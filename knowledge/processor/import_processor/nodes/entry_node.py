from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, PdfConversionError, \
    FileProcessingError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState
from dataclasses import dataclass
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class EntryNode(BaseNode):
    name="entry_node"

    def process(self,state: ImportGraphState):
        """
        根据上传文件后缀修改state中is_md_read_enabled is_pdf_read_enabled
        Returns:

        """
        #获得文件
        import_file_path_name=state.get("import_file_path","")
        file_dir=state.get("file_dir","")

        #判断文件是否存在
        if not import_file_path_name:
            raise StateFieldError(node_name=self.name, field_name="import_file_path",expected_type=str)
        import_file_path=Path(import_file_path_name)
        if not import_file_path.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path",expected_type=Path)

        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name="file_dir",expected_type=str)
        file_dir=Path(file_dir)
        if not file_dir.exists():
            raise StateFieldError(node_name=self.name, field_name="file_dir",expected_type=Path)

        #获得文件后缀
        if import_file_path.suffix ==".pdf":
            state["is_pdf_read_enabled"]=True
            state["pdf_path"]=str(import_file_path)
        elif import_file_path.suffix ==".md":
            state["is_md_read_enabled"]=True
            state["md_path"]=str(import_file_path)
        else:
            self.logger.error(f"此文件后缀格式{import_file_path.suffix}不支持")
            raise ValidationError(node_name=self.name,message=f"此文件后缀格式{import_file_path.suffix}不支持")

        #获得上传文件标题
        state["file_title"]=import_file_path.stem

        return state
if __name__ == '__main__':
    state=ImportGraphState(
        {"import_file_path":r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用\auto\万用表RS-12的使用_origin.pdf"},
        {"file_dir":r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用\auto"}
    )
    node=EntryNode()
    result=node.process(state)
    print(result)
