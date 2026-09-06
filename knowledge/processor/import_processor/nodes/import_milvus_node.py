import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Sequence

from langgraph.graph import END
from pymilvus import MilvusClient,DataType
from langgraph.graph.state import StateGraph, CompiledStateGraph
from dataclasses import dataclass

from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError,MilvusError
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PDFToMDNode
from knowledge.processor.import_processor.nodes.md_img_node import MDToImgNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.base import BaseNode
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.processor.import_processor.state import ImportGraphState

@dataclass
class _SCALAR_FIELD_SPC:
    """
    标量字段
    """
    field_name:str
    datatype:DataType
    max_length:Optional[int]= None
_SCALAR_FIELDS:Sequence[_SCALAR_FIELD_SPC]=(
    _SCALAR_FIELD_SPC(field_name="content",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535),
)



class _MilvusShameBuilder():
    """
    建造者1，负责处理milvus字段约束逻辑
    """
    @staticmethod
    def create_schema(milvus_client:MilvusClient,dim:int):
        """
        创建schema
        Args:
            self:
            milvus_client:
            dim:

        Returns:
        """
        #创建schema,设置为可添加动态字段
        schema=milvus_client.create_schema(enable_dynamic_field=True)

        #2.添加字段约束
        #2.1添加主键字段约束id,设置为自增,插入数据时无需插入该字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True,auto_id=True)
        #2.2添加向量字段约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
        #2.3添加标量字段约束
        for spec in _SCALAR_FIELDS:
            kwargs:Dict={
                "field_name":spec.field_name,
                "datatype":spec.datatype,
            }
            if spec.max_length:
                kwargs["max_length"]=spec.max_length
            schema.add_field(**kwargs)
        return schema

class _MilvusIndexBuilder():
    """
    建造者2，负责处理milvus索引逻辑
    """
    @staticmethod
    def create_index(milvus_client:MilvusClient):
        #1.创建索引
        index=milvus_client.prepare_index_params()
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        return index

class _InsertMilvusDataBuilder:
    """
    建造者3，负责添加数据行
    """
    @staticmethod
    def insert_raw(milvus_client:MilvusClient, chunks:List[Dict[str, Any]],collection_name:str):
        #插入数据
        insert_result=milvus_client.insert(collection_name=collection_name, data=chunks)
        #得到每一个chunk的id
        ids=insert_result.get("ids",[])
        #回填到chunk中
        for id,chunk in zip(ids,chunks):
            chunk["id"]=id



class MilvusImportNode(BaseNode):
    """
    门面，之对外保留process
    """
    name = "Milvus_import_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        Args:
            state:

        Returns:state

        """
        #1.校验参数
        chunks,dim=self._validate_state(state)
        #2.创建milvus客户端
        try:
            milvus_client=StorageClients.get_milvus()
        except Exception as e:
            raise MilvusError(message=f"Milvus客户端创建失败:{e}", node_name=self.name)

        #3.获得chunks集合名
        chunks_collection_name=self.config.chunks_collection
        #4.创建集合collection
        self._create_chunks_collection(milvus_client,chunks_collection_name,dim)
        #5.导入数据
        _InsertMilvusDataBuilder.insert_raw(milvus_client,chunks,chunks_collection_name)



    def _validate_state(self, state)->Tuple[List[Dict[str, Any]],int]:
        chunks=state.get("chunks")
        valid_chunks=[]
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(message=f"chunk_{index + 1}类型不匹配,实际类型为{type(chunk)}",
                                      node_name=self.name)
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                valid_chunks.append(chunk)
            else:
                self.logger.warning(f"chunk_{index + 1}缺少向量,跳过")

        if not valid_chunks:
            raise ValidationError(message="没有有效的向量,请检查", node_name=self.name)

        dim =len(valid_chunks[0]["dense_vector"])
        self.logger.info(f"有效chunk数{len(valid_chunks)}稠密向量维度为{dim}")
        return valid_chunks,dim

    def _create_chunks_collection(self, milvus_client:MilvusClient, chunks_collection_name:str,dim:int):
        #判断集合是否存在
        if milvus_client.has_collection(chunks_collection_name):
            self.logger.warning(f"集合{chunks_collection_name}已存在,跳过创建")
            return
        #2.创建约束schema
        schema=_MilvusShameBuilder.create_schema(milvus_client, dim)

        #3.创建索引
        index_params=_MilvusIndexBuilder.create_index(milvus_client)

        #4.创建集合collection
        try:
            collection=milvus_client.create_collection(
                collection_name=chunks_collection_name,
                schema=schema,
                index_params=index_params,
            )
        except Exception as e:
            raise MilvusError(message=f"集合{chunks_collection_name}创建失败:{e}", node_name=self.name)
        self.logger.info(f"集合{chunks_collection_name}创建成功")


def _cli_main() -> None:
    from knowledge.processor.import_processor.base import setup_logging
    import  json
    setup_logging()

    temp_dir = Path(r"D:\develop\develop\workspace\pycharm\usage\251208\knowledge\processor\import_processor\temp_dir"
    )
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content}

    node = MilvusImportNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")

if __name__ == '__main__':
    _cli_main()