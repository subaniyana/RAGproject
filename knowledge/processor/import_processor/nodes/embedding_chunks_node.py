import json
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any,Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from pymilvus import DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction#直接使用milvus提供的bge_m3
#from FlagEmbedding import  BGEM3FlagModel


from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, ConfigurationError, LLMError, \
    ValidationError, EmbeddingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

from knowledge.prompt.import_prompt import *


class EmbeddingChunksNode(BaseNode):
    name="Embedding_chunks_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        为每个文档切片生成向量
        Args:
            state:ImportGraphState
        Returns:
        """
        self.logger.info("开始处理文档切片向量")

        # 1.校验chunks字段有效性
        self.log_step(step_name="step1", message="先校验切片chunks状态")
        chunks=self._validate_state_chunks(state)

        #2.获得嵌入模型客户端
        self.log_step(step_name="step2", message="获得bge-m3嵌入模型客户端")
        try:
            embedding_client=AIClients.get_bge_m3()
        except Exception as e:
            self.logger.error(f"获得bge-m3嵌入模型客户端失败:{e}")
            raise EmbeddingError(message="获得bge-m3嵌入模型客户端失败",node_name=self.name)

        #3.批量嵌入
        self.log_step(step_name="step3", message="批量嵌入")
        #批量阈值
        batch_size=self.config.embedding_batch_size
        #chunk总数
        total=len(chunks)
        #遍历
        final_chunks=[]
        for index in range(0,total,batch_size):
            #当前这批次数据
            batch=chunks[index:index+batch_size]
            batch_end=index+len(batch)
            self.logger.info(f"处理第{index+1}-{batch_end}批次数据/{total}")

            batch_embeddings=self._embedding_batch(batch,embedding_client)
            final_chunks.extend(batch_embeddings)

        #更新state的chunks
        state["chunks"]=final_chunks
        return state

    def _validate_state_chunks(self, state: ImportGraphState)->List[Dict[str, Any]]:
        chunks = state.get("chunks")
        # 校验chunks字段是否存在且为list类型,其中每个元素是否是dict
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        for index,chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(message=f"chunk_{index+1}类型不匹配,实际类型为{type(chunk)}",node_name=self.name)
        return chunks

    def _embedding_batch(self, batch:List[Dict[str, Any]], embedding_client)->List[Dict[str, Any]]:
        """
        批量嵌入
        Args:
            batch:批量chunk数据
            embedding_client:嵌入模型客户端
        Returns:
            embeddings:嵌入向量列表
        """
        chunk_contents=[f"{chunk.get("content")}\n{chunk.get("item_name","")}" for chunk in batch]
        try:
            vectors=embedding_client.encode_documents(chunk_contents)
        except Exception as e:
            raise EmbeddingError(message="批量嵌入失败",node_name=self.name)
        if not vectors:
            raise EmbeddingError(message="批量结果不存在",node_name=self.name)
        sparse_ves=vectors.get("sparse")
        for i,chunk in enumerate(batch):
            chunk["dense_vector"]=vectors.get("dense")[i].tolist()
            chunk["sparse_vector"]=self._sparse_vector(sparse_ves,i)
        return batch

    def _sparse_vector(self, sparse_ves, i):
        """
        从稀疏矩阵中提取当前chunk对象的稀疏向量
        Args:
            sparse_ves:
            i:当前chunk

        Returns:
        """
        # 获得chunk的开始和解释索引
        start_index = sparse_ves.indptr[i]
        end_index = sparse_ves.indptr[i+1]
        # 获得token_id
        token_ids = sparse_ves.indices[start_index:end_index].tolist()
        # 获得weight
        weights = sparse_ves.data[start_index:end_index].tolist()
        # 构建稀疏向量字典{token_id:weight}
        sparse_vecs_dict = dict(zip(token_ids, weights))
        return sparse_vecs_dict

if __name__ == '__main__':
    from knowledge.processor.import_processor.base import setup_logging
    setup_logging()

    base_dir = Path(
        r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir"

    )
    input_path = base_dir / "chunks.json"
    output_path = base_dir / "chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = EmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至:\n{output_path}")