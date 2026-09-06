#内置包
import os
import subprocess,time,json
from pathlib import Path
from typing import Tuple
#三方包
import logging

#自己包
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, PdfConversionError
from knowledge.processor.import_processor.state import ImportGraphState


class PDFToMDNode(BaseNode):
    name="pdf_to_md_node"
    #ctrl+o:实现父类方法
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点处理逻辑入口
        :param state:
        :return:
        """
        #接收pdf文件path,利用mineru解析工具将pdf转化为md文件
        #校验传入参数
        import_path, output_dir = self._validate_state(state)

        #执行mineru解析文件为md文件(mineru -p inputpath -o output_dir --source=local)
        processed_status = self._execute_mineru(import_path, output_dir)
        if processed_status != 0:
            detail = getattr(self, "_mineru_output_tail", "未获取到 MinerU 的错误输出")
            raise PdfConversionError(
                message=(
                    f"Mineru解析失败（退出码: {processed_status}）。"
                    f"MinerU 最后输出：\n{detail}"
                ),
                node_name=self.name,
            )

        #获得md文件路径str
        md_path = self._get_md_path(import_path, output_dir)

        #添加md文件路径到状态中
        state["md_path"] = md_path

        return state

    def _validate_state(self, state: ImportGraphState)->Tuple[Path, Path]:
        """
        校验状态参数
        :param state:导入图谱节点状态
        :return:导入文件路径和输出目录
        Tuple[Path, Path]
        """
        self.log_step("step1","校验解析文件路径和输出文件路径是否合格")

        #判断文件路径是否存在
        if not state.get("import_file_path"):
            raise StateFieldError(node_name=self.name,field_name="import_file_path",expected_type=str)

        #获得导入文件路径
        import_path = Path(state["import_file_path"])

        #判断路输入路径是否有效
        if not import_path.exists():
            raise StateFieldError(node_name=self.name,field_name="import_file_path",expected_type=str,message=f"导入文件路径不存在: {import_path}")

        #获得输出文件目录
        output_dir = state.get("file_dir", "")

        #判断输出文件目录是否存在,不存在则以导入文件亲目录为默认目录
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = import_path.parent

        #判断是否是一个有效目录
        if not output_dir.exists():
            raise StateFieldError(node_name=self.name,field_name="output_dir_path",expected_type=str,message=f"输出文件目录不存在: {output_dir}")

        self.logger.info(f"解析文件路径: {import_path}")
        self.logger.info(f"输出文件目录: {output_dir}")
        return import_path, output_dir

    def _execute_mineru(self, import_path: Path, output_dir: Path)-> int:
        """
        执行mineru解析文件为md文件
        :param import_path:导入文件路径
        :param output_dir:输出文件目录
        :return:状态【0 或 非0】
        0：成功，非0：失败
        """
        self.log_step("step2","执行mineru解析文件为md文件")
        #执行mineru解析文件为md文件
        #mineru -p inputpath -o output_dir --source=local
        #定义cmd
        cmd=[
            self.config.mineru_url,
            # r"D:\aloud\softapp\anaconda1\envs\py312RAG\Scripts\mineru.exe",
            "-p",
            str(import_path),
            "-o",
            str(output_dir),
            "--source", "local",
            "-b", "pipeline"
        ]

        #开始记时
        start_time = time.time()

        #利用子进程执行cmd,并获得子进程解析的日志，子进程产生日志发通过管子给外部进程
        output_tail = []
        proc=subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,#接收正确日志
                         stderr=subprocess.STDOUT,#接收错误日志
                         text=True,#输出日志为文本
                         errors="replace",#特殊字符串码替换为？
                         encoding="utf-8",#编码格式
                         bufsize=1,#按行输出
                         env=os.environ.copy(),#将 .env 已加载的本地模型配置传递给子进程
                         )

        #打印解析日志
        for line in proc.stdout:
            output_tail.append(line.rstrip())
            # 只保留最后 60 行，既保留根因，又避免异常信息过长。
            if len(output_tail) > 60:
                output_tail.pop(0)
            self.logger.info(f"Mineru解析日志: {line}")

        #主进程等待子进程结束
        process_status=proc.wait()
        self._mineru_output_tail = "\n".join(output_tail) or "未产生标准输出。"

        #结束记时
        end_time = time.time()
        elapsed_time = end_time - start_time
        if process_status==0:
            self.logger.info(f"Mineru解析成功，耗时: {elapsed_time:.2f}秒")
        else:
            self.logger.error( f"Mineru解析失败，h耗时: {elapsed_time:.2f}秒")
        return process_status

    def _get_md_path(self, import_path:Path, output_dir: Path)-> str:
        """
        获得md文件路径
        :param import_path:导入文件路径
        :param output_dir:输出文件目录
        :return:md文件完整路径
        """
        #path对象：name 文件名+后缀名，stem 文件名，suffix 后缀名
        self.log_step("step3","获得md文件路径")
        #获得md文件路径
        file_name = import_path.stem
        file_name = import_path.stem
        target_dir = output_dir / file_name

        # 动态搜索该目录下生成的所有 .md 文件（兼容 hybrid_auto、auto 等不同模式的输出路径）
        md_files = list(target_dir.rglob("*.md"))
        if md_files:
            return str(md_files[0])

        # 兜底路径
        return str(target_dir / "auto" / f"{file_name}.md")

if __name__ == '__main__':
    from knowledge.processor.import_processor.base import setup_logging
    setup_logging(logging.INFO)  # 配置日志

    pdf_to_md_node = PDFToMDNode()
    init_state = {
        "import_file_path": "D:\\文件\\LLMLearn\\17langchain大项目\\pdf文档\\doc\\万用表RS-12的使用.pdf",
        "file_dir": "E:\\LLMlearn\\daima\\RAGproject\\knowledge\\processor\\import_processor\\temp_dir"
    }
    result=pdf_to_md_node.process(init_state)

    #序列化：将对象转换为字符串
    result_str=json.dumps(result,indent=4,ensure_ascii=False)
    print(f"解析结果: {result_str}")
