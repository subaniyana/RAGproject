# 内置包
import base64
import re
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Deque
import logging

# 三方包
from langchain_openai import OpenAI

# 自己包
from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, PdfConversionError, \
    FileProcessingError, ImageProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from dataclasses import dataclass
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass  # 数据类，直接实例化对象，不用重新init等方法
class ImageContext:
    """
    图片上下文
    """
    head: str  # 上文标题
    pre_text: str  # 上文内容
    tail: str  # 下文内容


@dataclass
class ImageInfo:
    """
    一张图片的完整信息
    图片名字：作为图片摘要的字典key
    地址：1.vlm调用图片接口 2.本地存储
    图片上下文：给VLM模型提供图片上下文，帮助模型解析图片
    """
    img_context: ImageContext  # 图片上下文
    name: str  # 图片名称
    path: str  # 图片路径


class _MdFileHander:
    """
    作用：
    1.读取md内容，md_path,图片目录
    2.备份新的md_content
    """

    def __init__(self, logger, node_name: str):
        self.node_name = node_name
        self.logger = logger

    def valid_read_md_file(self, state) -> Tuple[str, Path, Path]:
        """
        作用
        读取md内容，md_path,图片目录
        Args:
            state:上一个节点更新后的state
        Returns:
            Tuple[str, Path, Path]
        """

        # 获得md文件路径
        md_path = state.get("md_path", "")
        if not md_path:
            raise StateFieldError(node_name=self.name, field_name="md_path", expected_type=str)

        # path标准化
        md_path_obj = Path(md_path)

        # 读取md_content内容
        try:
            with md_path_obj.open("r", encoding="utf-8") as f:
                md_content = f.read()
        except Exception as e:
            self.logger.error(f"{self.node_name} 读取md文件失败: {md_path_obj}")
            raise FileProcessingError(message="读取md文件失败", node_name=self.name, cause=e)

        # 获得图片目录
        img_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, img_dir

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class _ImageScanner:
    """
    作用：
    1.根据图片目录得到有效的图片文件
    2.定位md文件中此图片的位置
    3.获得此图片再md文件中的上下文（给VLM模型提供图片上下文，帮助模型解析图片）
    4.组装所有图片上下文信息
    """

    def __init__(self, logger):
        self.logger = logger

    def scan_imgs_dir(self, img_dir: Path, md_content: str, image_extensions: Set[str], img_content_length: int) -> \
    List[ImageInfo]:
        """
        作用：
        1.根据图片目录得到有效的图片文件
        2.定位md文件中此图片的位置
        2.1上文信息（标题和上文内容）
        2.2下文信息（下文内容）
        3.组装所有图片上下文信息封装到图片所有信息的容器中info_list
        Args:
            img_dir:图片目录
            md_content:md内容
            image_extensions:合法文件类型后缀
            img_content_length:图片上下文各自长度

        Returns:List[ImageInfo]所有图片全部信息
        """
        img_info_list = []
        # 遍历图片目录
        for img_path in img_dir.iterdir():

            # 过滤子目录
            if not img_path.is_file():
                self.logger.info(f"{img_path}不是一个文件")
                continue
            # 过滤不合法的文件
            if not img_path.suffix in image_extensions:
                self.longger.errror(f"{img_path}文件后缀类型不合法")
                continue
            # 查找图片上下文
            ctx: ImageContext = self._find_context(img_path.name, md_content, img_content_length)
            if not ctx:
                self.logger.info(f"{img_path}没有上下文")
                continue
            # 封装图片信息
            img_info_list.append(ImageInfo(
                img_context=ctx,
                name=img_path.name,
                path=str(img_path)))
        self.logger.info(f"md中找到多少个图片数量为：{len(img_info_list)}")
        return img_info_list

    def _find_context(self, img_name: str, md_content: str, img_content_length: int) -> Optional[ImageContext]:
        """
        作用：
        根据图片名称，在md内容中查找图片上下文
        Args:
            img_name:图片名称
            md_content:md内容
            img_content_length:图片上下文长度

        Returns:找到返回imagecontext
                没找到返回none
        """
        self.logger.info(f"开始查找图片上下文，图片名称为：{img_name}")
        lines = md_content.split('\n')
        # 选择适用的正则（例：标准Markdown）
        pattern = re.compile(rf'!\[[^\]]*\]\([^)]*?(?:^|\/|\\){re.escape(img_name)}(?:\.[a-zA-Z]+)?[^)]*\)')

        for line_idx, line in enumerate(lines):
            if not pattern.search(line):
                continue
            # 上文
            head, prev_index = self._find_head(line_idx, lines)
            pre_lines = lines[prev_index + 1:line_idx]
            pre_context = self._lines_to_context(pre_lines, img_content_length, direction="pre")
            # 下文
            next_index = self._find_next(line_idx, lines)
            next_lines = lines[line_idx + 1:next_index]
            next_context = self._lines_to_context(next_lines, img_content_length, direction="next")

            return ImageContext(
                head=head,
                pre_text=pre_context,
                tail=next_context)
        return None

    def _find_head(self, line_idx, lines):
        """
        作用：

        Args:
            line_idx: 图片索引号
            lines: 整个md的line

        Returns:返回当前图片的上文标题的内容和索引
        """
        for i in range(line_idx - 1, -1, -1):
            if re.match(r'^#{1,6}\s+(.+)$', lines[i]):
                return lines[i], i
        return "", -1

    def _find_next(self, line_idx, lines):
        """
        作用：

        Args:
            line_idx: 图片索引号
            lines: 整个md的line

        Returns:返回当前图片的下文标题的索引
        """
        for i in range(line_idx + 1, len(lines)):
            if re.match(r'^#{1,6}\s+(.+)$', lines[i]):
                return i
        return len(lines)

    def _lines_to_context(self, pre_or_next_lines, img_content_length, direction) -> str:
        """
        作用：截取指定上下文内容
        截取策略：根据段落截取，根据段落字符数是否达到最大上下文来截取
        当lines间出现\n或者图片则视为一个段落分割
        Args:
            pre_or_next_lines: 上（下）文
            img_content_length:上下文长度
            direction: 方向("pre","next")

        Returns:返回上下文内容
        """
        paragraphs = []
        current_paragraph = []
        for line in pre_or_next_lines:
            # 是空白行
            is_blank_line = not line.strip()
            # 是图片行
            is_image_line = re.match(r'!\[[^\]]*\]\([^)]*\)', line)
            # 当此行是空白或者图片行，则将当前段落加入段落列表，并清空当前段落，继续下一行
            if is_blank_line or is_image_line:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue
            current_paragraph.append(line)
        # 添加最后的几行
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 针对方向为"pro"的调用进行翻转,使优先保留离图片近的段落
        if direction == "pre":
            paragraphs.reverse()

        # 遍历段落，限制字数
        count = 0
        selected_paragraphs = []
        for paragraph in paragraphs:
            if count + len(paragraph) <= img_content_length or not selected_paragraphs:
                selected_paragraphs.append(paragraph)
                count += len(paragraph)
            else:
                break
        # 将段落翻转回来，使段落顺序正常
        if direction == "pre":
            selected_paragraphs.reverse()
        # 将段落列表转为字符串
        return "\n\n".join(selected_paragraphs)


class _VLMSummarizer:
    """
    作用：
    根据图片上下文信息，调用vlm模型，得到图片摘要
    """

    def __init__(self, logger, requests_per_minite: int):
        self.requests_per_minite = requests_per_minite
        self.logger = logger
        self.max_image_side = 512
        self.image_quality = 70

    def _summarize_img_to_context(self, document_name, img_info_list, vl_model) -> Dict[str, str]:
        """
        作用：根据图片信息
        Args:
            img_info_list: 图片全信息列表
            vl_model: 模型名称

        Returns:{"img_name":"summary"}
        """
        summaries = {}
        requests_timestamp: Deque[float] = deque()
        # 获得vlm客户端
        try:
            vlm_client = AIClients.get_openai()
        except Exception as e:
            self.logger.error(f"获取openai客户端失败")
            for img_info in img_info_list:
                summaries[img_info.name] = "图片摘要生成失败"
            return summaries

        # 调用vlm生成摘要
        for img_info in img_info_list:
            self._enforce_rate_limit(requests_timestamp, self.requests_per_minite)
            try:
                summaries[img_info.name] = self._summarize_one(document_name, img_info, vlm_client, vl_model)
            except Exception as e:
                self.logger.error(f"图片{img_info.name}摘要生成失败:{e}")

        return summaries

    def _summarize_one(self, document_name: str, img_info: ImageInfo, vlm_client: OpenAI, vl_model: str) -> str:
        """
        作用：用vlm模型得到一张图片的摘要
        Args:
            document_name:文件名
            img_info: 图片信息
            vlm_client: vlm客户端
            vl_model: vlm模型名称

        Returns:图片摘要
        """
        # 获得图片上下文（标题，上文，下文）
        context = [p for p in (img_info.img_context.head, img_info.img_context.pre_text, img_info.img_context.tail) if
                   p]
        final_context = "\n".join(context) if context else "暂无上下文"

        # 根据图片地址获得图片内容，图片是二进制数据读取，base64编码为bytes,utf-8解码为可传输的str
        try:
            img_data = self._encode_image_for_vlm(img_info.path)
        except Exception as e:
            self.logger.error(f"图片内容获取失败:{e}")
            return "暂无图片描述"

        # 利用vlm客户端调用vlm
        try:
            start_time = time.time()
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"任务：为Markdown文档中的图片生成简短中文标题。\n"
                                    f"Markdown文档标题：\"{document_name}\"\n"
                                    f"图片上下文：{final_context}\n"
                                    f"请结合图片内容和上下文信息。\n"
                                    f"用概括这张图片内容，生成一个精准且25字以内的中文标题摘要。（不包含图片两字）\n"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                        }
                    ]
                }],
                # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
                extra_body={
                    'enable_thinking': False,
                    "thinking_budget": 81920},
                # max_tokens=32,
                # temperature=0.3,
                # top_p=0.9,
                # timeout=30  # 加超时控制
            )
            self.logger.info(f"图片{img_info.name} VLM调用耗时: {time.time() - start_time:.2f}秒")
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败:{e}")
            return "暂无图片描述"

    def _encode_image_for_vlm(self, img_path: str) -> str:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _enforce_rate_limit(self, timestamps: Deque[float], max_requests: int, time_window_size: int = 60):
        """
        强制执行速率限制，如果请求频率超过限制则主动等待。
        该函数使用滑动窗口算法，维护一个请求时间戳队列，当窗口内请求数达到上限时，
        计算需要等待的时间并执行 sleep，直到有可用配额。

        Args:
            timestamps: 存储请求时间戳的双端队列（按时间升序排列），
                        会被本函数修改（移除过期时间戳并追加当前时间戳）。
            max_requests: 时间窗口内允许的最大请求数。
            time_window_size: 时间窗口大小，单位：秒，默认 60 秒。
        Returns:
            None
        """
        now = time.time()
        # 移除窗口外或者超过 window 秒前的旧时间戳
        while timestamps and now - timestamps[0] > time_window_size:
            timestamps.popleft()

        # 检查是否已达到请求上限
        if len(timestamps) >= max_requests:
            sleeptime = time_window_size - (now - timestamps[0])
            if sleeptime > 0:
                self.logger.info(
                    f"达到速率限制（{max_requests}窗口达到上限，暂停 {sleeptime:.2f} 秒..."
                )
                time.sleep(sleeptime)
        now = time.time()

        # 再次清除旧时间戳
        while timestamps and now - timestamps[0] > time_window_size:
            timestamps.popleft()
        timestamps.append(now)


class _ImageUploader:
    """
    作用：
    1.上传图片到minio图片存储服务,得到远程地址
    2.将md中摘要添加和图片地址更新
    """

    def __init__(self, logger):
        self.logger = logger

    def upload_and_replace(self, md_content: str, stem: str, img_info_list: List[ImageInfo], summarices: Dict[str, str],
                           minio_url: str, minio_bucket_name: str) -> str:
        """
        上传文件和图片到minio并且跟新md中图片地址以及摘要
        Args:
            md_content:         md内容
            stem:               md名字作为minio对象目录
            img_info_list:      图片信息
            summarices:         图篇及其摘要{str:str}
            minio_url:          minio地址
            minio_bucket_name:  minio桶名

        Returns:跟新后的md内容
        """
        # 上传图片
        remote_urls = self._upload_images(stem,img_info_list, minio_url, minio_bucket_name)
        #更新md中图片地址
        md_content:str= self._update_md(md_content, remote_urls, summarices)
        return md_content

    def _upload_images(self, dir_stem, img_info_list, minio_url, minio_bucket_name):
        #得到minio客户端
        remote_urls = {}
        try:
            minio_client = StorageClients.get_minio()
        except Exception as e:
            for img_info in img_info_list:
                remote_urls[img_info.name]=img_info.path
            return remote_urls

        #遍历上传每一个图片
        for img_info in img_info_list:
            object_name=f"{dir_stem}/{img_info.name}"
            #上传图片到minio
            try:
                minio_client.fput_object(
                    minio_bucket_name,object_name,img_info.path)
                self.logger.info(f"图片{img_info.name}上传minio成功:")
                #拼装路径http://192.168.100.120:9000/桶名/对象名
                remote_urls[img_info.name]=f"{minio_url}/{minio_bucket_name}/{object_name}"
            except Exception as e:
                self.logger.error(f"图片{img_info.name}上传minio失败转用本地图片地址{img_info.path}")
                remote_urls[img_info.name]=img_info.path
        return remote_urls

    def _update_md(self, md_content:str, remote_urls:Dict[str, str], summarices:Dict[str, str]):
        """
        更新md中的图片描述和远程图片地址
        Args:
            md_content: 原md
            remote_urls:minio生成的url
            summarices:vlm生成的摘要

        Returns:新的md
        """
        pattern=re.compile(r"!\[(.*?)\]\((.*?)\)")
        def replace_func(match)->str:
            """
            替换规则
            Args:
                match:正则表达式子
            Returns:![摘要](图片地址)
            """
            img_path = match.group(2)
            img_name_in_md = Path(img_path).name
            # 遍历图片摘要{图片名，摘要}，当图片名与正则表达式匹成功的图片名一致，将匹配图片替换摘要与远端图片地址
            for img_name, img_summary in summarices.items():
                if img_name == img_name_in_md:
                    return f"![{img_summary}]({remote_urls.get(img_name, img_path)})"
            return match.group(0)
        return pattern.sub(replace_func,md_content)





class MDToImgNode(BaseNode):
    """
    作用：
    1.得到四个类实例对象
    2.调用对象处理方法
    """
    name = "md_to_img_node"

    def __init__(self):
        super().__init__()
        self._md_file_hander = _MdFileHander(self.logger, self.name)
        self._image_scanner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._image_uploader = _ImageUploader(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑
        ：param state:
        ：return:
        """
        config = self.config
        # 1.操作_md_file_hander
        self.log_step("step1", "操作_md_file_hander,读取md内容，md_path,图片目录")
        md_content, md_path_obj, img_dir = self._md_file_hander.valid_read_md_file(state)

        # 判断是否有图片，无图片则直接返回
        if not img_dir.exists():
            state["md_content"] = md_content
            return state

        # 2.操作_image_scanner
        self.log_step("step2", "操作_image_scanner,扫描图片目录")
        img_info_list: List[ImageInfo] = self._image_scanner.scan_imgs_dir(img_dir,
                                                                           md_content,
                                                                           config.image_extensions,
                                                                           config.img_content_length)

        # 3.操作_vlm_summarizer
        self.log_step("step3", "操作_vlm_summarizer,生成图片摘要")
        summarices: Dict[str, str] = self._vlm_summarizer._summarize_img_to_context(md_path_obj.stem, img_info_list,
                                                                                    config.vlm_model)
        print(summarices)
        # 4.操作_image_uploader
        self.log_step("step4", "操作_image_uploader,上传图片到minio,并且得到新的md")
        new_md_content=self._image_uploader.upload_and_replace(md_content,
                                                md_path_obj.stem,
                                                img_info_list,
                                                summarices,
                                                config.get_minio_base_url(),
                                                config.minio_bucket
                                                )
        #备份新的md文件
        self.log_step("step5", "备份新的md文件")
        self._md_file_hander.backup(md_path_obj,new_md_content)
        state["md_content"] = new_md_content
        return state


if __name__ == '__main__':
    from knowledge.processor.import_processor.base import setup_logging

    setup_logging(logging.INFO)  # 配置日志
    md_img_node = MDToImgNode()
    init_state = {
        "md_path": "E:\\LLMlearn\\daima\\RAGproject\\knowledge\\processor\\import_processor\\temp_dir\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md"
    }
    result = md_img_node.process(init_state)
