import json
import os
import re
from typing import Tuple, List, Dict, Any
from langchain_text_splitters import  RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """
    文档切分节点
    """
    name="document_split_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分核心逻辑入口
        Args:
            state:

        Returns:

        """
        #1.参数校验
        config = self.config
        md_content, file_title, max_content_length, min_content_length=self._validate_state(state,config)

        #2.切分（一级：根据md文档中的标题来切分）内容
        sections:List[Dict[str,Any]]=self._split1_by_headings(md_content,file_title)

        # 调试：打印一级切分结果
        self.logger.info(f"一级切分后 section 数量: {len(sections)}")
        for idx, sec in enumerate(sections):
            self.logger.info(f"  section[{idx}] title='{sec['title']}', parent_title='{sec['parent_title']}', body_len={len(sec['body'])}")

        #3.切分（二级：根据md文档中的内容来切分并合并）内容
        sections:List[Dict[str,Any]]=self._split2_by_content(sections,max_content_length,min_content_length)

        # 调试：打印二级切分结果
        self.logger.info(f"二级切分后 section 数量: {len(sections)}")
        for idx, sec in enumerate(sections):
            self.logger.info(f"  section[{idx}] title='{sec['title']}', parent_title='{sec['parent_title']}', body_len={len(sec['body'])}")

        #4.组装成chunk对象
        chunks:List[Dict[str,Any]]=self._assemble_chunks(sections)

        #5.备份
        self._backup_chunks(chunks,state)

        #6.更新state
        state["chunks"]=chunks

        #7.返回
        return state



        pass
    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:
        """
        从状态字典中获取 Markdown 内容和配置参数，统一换行符格式，校验参数合法性。
            - Windows 换行符 \r\n 转为 \n
            - Mac 旧版换行符 \r 转为 \n
        得到文件md_content, 源文件标题file_title
        - max_content_length 和 min_content_length 必须大于 0
        - max_content_length 必须大于 min_content_length
        Args:
            state:ImportGraphState
            config:BaseNode的配置文件
        Returns:md_content, file_title, config.max_content_length, config.min_content_length
        """

        self.log_step("step1", "切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
           or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split1_by_headings(self, md_content, file_title)->List[Dict[str,Any]]:
        """
        根据标题#{1-6}来对文档进行第一层切分，_flush()用于在sections中添加段落
        正则表达式匹配标题->匹配成功对之前存储的段落刷新->没有匹配到的句子进入待刷新段落
        Args:
            md_content: md内容
            file_title: 文件标题

        Returns:List[Dict[str,Any]]切分后多个章节
        """
        in_fence=False
        body_lines=[]
        sections=[]#最终搜集到的分段内容{普通行，标题，父标题，文档标题}
        current_title=""
        hierarchy=[""]*7#存储所有标题
        current_level=None
        # 切分每行
        md_lines = md_content.split("\n")

        def _flush():
            """
            打包section
                    {body:收集到底所有行
                     title：当前内容的标题
                     parent——title:当前内容父标题
                     file_title:文档标题}
            Returns:
            """
            #处理内容行,刷新sections内容
            body="\n".join(body_lines)
            if current_title or body:
                parent_title=""
                if current_level is not None:
                    for i in range(current_level-1,0,-1):
                        if hierarchy[i]:#父标题不能为空
                            parent_title = hierarchy[i]
                            break
                #没有上层标题情况：1.处理两个标题连载一起 2.文章开头没有标题的情况
                if not parent_title:
                    parent_title=current_title if current_title else file_title
                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,
                    "parent_title": parent_title,
                    "file_title": file_title
                })

        #定义正则
        pattern=re.compile(r"^\s*(#{1,6})\s+(.+)")
        for md_line in md_lines:
            #检查代码围栏~~~ or ```
            if md_line.startswith("```") or md_line.startswith("~~~"):
                in_fence= not in_fence
            match=pattern.match(md_line) if not in_fence else None
            if match:
                #将body_lines中的内容都添加到sections中
                _flush()
                body_lines.clear()  # 清空已刷新的行，避免累积

                current_title=match.group(2).strip()#当前标题（去掉##前缀）
                level=len(match.group(1))#当前标题层级
                current_level=level
                hierarchy[level]=current_title

                #清空不属于自己的子标题
                for i in range(level+1,7):
                    hierarchy[i]=""
            else:
                body_lines.append(md_line)

        _flush()#添加最后几行
        return sections

    def _split2_by_content(self, sections, max_content_length, min_content_length)->List[Dict[str,Any]]:
        """
        根据内容来对文档进行第二层切分较大章节，合并较小章节
        Args:
            sections:第一层切分后的章节
            max_content_length:最大内容长度（title+body）
            min_content_length:最小内容长度(title+body)

        Returns:List[Dict[str,Any]]切分后多个章节
        """
        #切分段落
        current_sections=[]
        for section in sections:
            current_section=self._split_long_section(section,max_content_length)
            current_sections.extend(current_section)

        #合并小章节
        sections_final=self._merge_short_sections(current_sections,min_content_length)

        return sections_final

    def _split_long_section(self, section, max_content_length)->List[Dict[str,Any]]:
        """
        按照max来切分长的章节内容
        Args:
            section: 当前章节
            max_content_length:最大内容长度（title+body）

        Returns:list[section]
        """
        body=section.get("body")
        title=section.get("title")
        parent_title=section.get("parent_title")
        file_title=section.get("file_title")

        #检查body中是否有表格
        if "<table>" in body:
            self.logger.info("包含表格，进行表格处理")
            body=MarkdownTableLinearizer.process(body)
        title_len=len(f"{title}\n\n")
        body_len=len(body)
        #判断总长度是否超过阈值，没超过则直接返回
        if (title_len+body_len)<=max_content_length:
            return [section]
        #防止是由于标题过于长的情况而分割
        body_max_len=max_content_length-title_len
        if body_max_len<0:
            return [section]
         #使用langchain的切分器进行切分，创建切分器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_max_len,
            chunk_overlap=0,
            separators=["\n\n", "\n","。","？","！","；",".","?","!",";",",", " ", ""],
            keep_separator=True
        )

        #切分器对象切分
        sections=text_splitter.split_text(body)
        #遍历
        sub_sections=[]
        for index, section in enumerate(sections):
            sub_section={
                "body": section,
                "title": f"{title}_{index+1}",
                "parent_title": parent_title,
                "file_title": file_title
            }
            sub_sections.append(sub_section)
        return sub_sections

    def _merge_short_sections(self, current_sections, min_content_length)->List[Dict[str,Any]]:
        """
        贪心累加算法：将同源的短 section 合并
        Args:
            current_sections: 所有sections列表
            min_content_length: 最小section的长度

        Returns:
            合并后的同源section对象们
                同源：属于同一个父标题
                合并后长度超过了min
        """
        # 1. 初始化
        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历合并
        for next_section in current_sections[1:]:
            same_parent = (current_section['parent_title'] == next_section['parent_title'])

            if same_parent and len(current_section.get('body')) < min_content_length:
                # 合并 body
                current_section['body'] = (
                        current_section.get('body').rstrip() + "\n\n" + next_section.get('body').lstrip()
                )
                # 仅当存在真正的父标题（非文件名）时，标题才回退为父标题
                if current_section['parent_title'] != current_section['file_title']:
                    current_section['title'] = current_section['parent_title']
            else:
                # 封箱
                final_sections.append(current_section)
                current_section = next_section

        # 最后一个封箱
        final_sections.append(current_section)

        return final_sections

    def _assemble_chunks(self, sectionsList:[Dict[str,Any]])->List[Dict[str,Any]]:
        """
        将sections组装成chunks{content,title,parent_title,file_title}
        Args:
            sections:切分后的章节列表
        Returns:
            chunks:文档切片列表
        """
        final_chunks=[]
        for section in sectionsList:
            body=section.get("body")
            title=section.get("title")
            parent_title=section.get("parent_title")
            file_title=section.get("file_title")
            content=f"{title}\n\n{body}"
            final_chunks.append({
                "content":content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title
            })
        self.logger.info(f"开始组装文档切片，切片数量chunk为：{len(final_chunks)}")
        return final_chunks

    def _backup_chunks(self, chunks:List[Dict[str,Any]],state):
        """将切分结果备份到 JSON 文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)#目录存在，也不报错
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    import logging
    from knowledge.processor.import_processor.base import setup_logging
    from knowledge.processor.import_processor.config import ImportConfig
    from knowledge.processor.import_processor.state import create_default_state

    setup_logging(logging.INFO)

    # ========== 辅助函数 ==========
    def make_node(max_len=1000, min_len=500):
        config = ImportConfig(max_content_length=max_len, min_content_length=min_len)
        return DocumentSplitNode(config=config)

    def make_state(md_content, file_title="测试文档", **kwargs):
        return create_default_state(md_content=md_content, file_title=file_title, **kwargs)

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        global passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}  {detail}")


    node = make_node()



    # 6.3 使用实际md文件
    print("\n===== 测试6.3: 实际md文件端到端 =====")
    md_path = r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用\auto\万用表RS-12的使用_new.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            real_md = f.read()
        st = make_state(real_md, file_title="万用表RS-12的使用",
                        file_dir=r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir")
        result = node.process(st)
        chunks = result["chunks"]
        check("实际md-chunks非空", len(chunks) >= 1)
        for i, c in enumerate(chunks):
            print(f"    chunk[{i}] title='{c['title']}', content_len={len(c['content'])}")
        titles = [c["title"] for c in chunks]
        check("实际md-title不全为文件名",
              any(t != "万用表RS-12的使用" for t in titles),
              f"titles={titles}")
    else:
        print(f"  [SKIP] md文件不存在: {md_path}")

    # ========== 汇总 ==========
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"测试完成: {passed}/{total} 通过, {failed}/{total} 失败")
    if failed == 0:
        print("全部通过！")
    else:
        print(f"有 {failed} 个测试失败，请检查。")
    print(f"{'='*50}")
