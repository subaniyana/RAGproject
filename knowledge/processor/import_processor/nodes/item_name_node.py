
from typing import Tuple, List, Dict, Any,Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType
from knowledge.utils.client.storage_clients import StorageClients
from pymilvus.model.hybrid import BGEM3EmbeddingFunction#直接使用milvus提供的bge_m3
#from FlagEmbedding import  BGEM3FlagModel


from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, ConfigurationError, LLMError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.prompt.import_prompt import *


class ItemNameRecognitionNode(BaseNode):
    """
    1.利用llm提取商品名
    2.嵌入商品具体信号
    3.存储到milvus中
    提取商品名
    """
    name = "Item_name_recognize_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        商品名提取核心逻辑入口

        """

        #校验参数
        file_title,chunks,item_name_chunk_k,item_name_chunk_size=self._check_state_field(state)
        #构建商品名识别上下文
        item_name_context=self._prepare_lls_context(chunks,item_name_chunk_k,item_name_chunk_size)
        #调用llm模型进行商品名识别
        item_name=self._use_llm_to_recognize_item_name(item_name_context,file_title)
        #将llm提取到商品名进行向量化嵌入
        dense_vecs,lexical_weights=self._embedding_item_name(item_name)
        
        #存储到milvus
        self._insert_milvus(file_title,item_name,dense_vecs,lexical_weights)

        #回填更新llm提取到的item_name
        self._fill_item_name(state,item_name,chunks)
        return  state


    def _check_state_field(self, state)->Tuple[str,List[str],int,int]:
        """
        校验状态字段
        Args:
            state:
        Returns:file_title,chunks,item_name_chunk_k,item_name_chunk_size
        """

        file_title=state.get("file_title")
        if not file_title:
            raise StateFieldError(node_name=self.name,field_name="file_title",expected_type=str)

        chunks=state.get("chunks")
        if not chunks:
            raise StateFieldError(node_name=self.name,field_name="chunks",expected_type=List[str])

        #设置商品名识别时使用切片数量，与切片长度
        item_name_chunk_k=self.config.item_name_chunk_k
        item_name_chunk_size=self.config.item_name_chunk_size

        if not item_name_chunk_k or item_name_chunk_k<=0:
            raise ConfigurationError(node_name=self.name,message="配置item_name_chunk_k错误")
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ConfigurationError(node_name=self.name,message="配置item_name_chunk_size错误")

        return file_title,chunks,item_name_chunk_k,item_name_chunk_size

    def _prepare_lls_context(self, chunks, item_name_chunk_k, item_name_chunk_size):
        """
        准备chunks商品名上下文
        Args:
            chunks:
            item_name_chunk_k:
            item_name_chunk_size:

        Returns:上下文信息

        """
        final_context=[]
        for index,chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue
            context=chunk.get("content") or chunk.get("context")
            split_context=f"{index}_{context}"
            final_context.append(split_context)
        return "\n".join(final_context)

    def _use_llm_to_recognize_item_name(self, item_name_context:str,file_title:str)->str:
        """
        使用llm模型进行商品名识别
        Args:
            item_name_context:商品名上下文
            file_title:文件标题

        Returns:商品名

        """
        try:
            llm_client=AIClients.get_llm_openai(resonse_format=False)
        except Exception as e:
            self.logger.error(f"获取chatopenai_llm客户端失败:{e}")
            raise LLMError(message="chat_open_ai llm获得失败")
        #调用llm模型进行商品名识别
        #1.获得商品名llm系统提示词
        system_prompt=ITEM_NAME_SYSTEM_PROMPT
        #2.获得商品名llm用户提示词
        user_prompt=ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title,context=item_name_context)
        #3.调用llm模型进行商品名识别
        try:
            llm_responce=llm_client.invoke(
                [SystemMessage(content=system_prompt),
                 HumanMessage(content=user_prompt)]
            )
            llm_result=llm_responce.content.strip()
            self.logger.info(f"文档{file_title}用llm商品名识别结果:{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"调用chat_open_ai llm模型失败:{e}")
            raise LLMError(message="chat_open_ai llm调用失败")

    def _embedding_item_name(self, item_name:str)->Tuple[Optional[List],Optional[Dict[str,Any]]]:
        """
        商品名嵌入模型
        Args:
            item_name:

        Returns:

        """
        try:
            bge_m3_client=AIClients.get_bge_m3()
        except Exception as e:
            self.logger.error(f"获取bge_m3客户端失败:{e}")
            return None,None
        try:
            # 解析稠密向量和稀疏向量
            vector_result=bge_m3_client.encode_documents(documents=[item_name])
            #稠密
            dense_vecs=vector_result.get("dense")[0].tolist()
            #稀疏
            sparse_vecs=vector_result.get("sparse")
            #从稀疏矩阵中提取第一个样本的数据开始结束索引
            start_index=sparse_vecs.indptr[0]
            end_index=sparse_vecs.indptr[1]
            #获得token_id
            token_ids=sparse_vecs.indices[start_index:end_index].tolist()
            #获得weight
            weights=sparse_vecs.data[start_index:end_index].tolist()
            #构建稀疏向量字典{token_id:weight}
            sparse_vecs_dict=dict(zip(token_ids,weights))
            self.logger.info(f"商品名{item_name}向嵌入向量成功")
            return dense_vecs,sparse_vecs_dict
        except Exception as e:
            self.logger.error(f"商品名{item_name}向嵌入向量失败:{e}")
            return None,None

    def _insert_milvus(self, file_title,item_name,  dense_vecs, sparse_vecs):
        """
        将llm识别到的商品名保存到milvs
        row记录{dense_vecs:,sparse_vector,file_title,item_name}
        Args:
            file_title:
            item_name:
            dense_vecs:
            sparse_vecs:

        Returns:
        """
        # 检查向量维度是否都存在
        if not dense_vecs or not sparse_vecs:
            self.logger.error(f"商品名{item_name}向嵌入向量失败")
            return
        # 获得milvus客户端
        try:
            milvus_client=StorageClients.get_milvus()
        except Exception as e:
            self.logger.error(f"获取milvus客户端失败:{e}")
            return
        #Milvus(集合三大核心【集合名，字段约束schema，索引】)
        #创建集合（约束，索引）-》插入数据
        #1.集合名：类似表名
        #2.约束：类似字段长度，类型
        #3.索引：提高查找效率
        item_name_collection=self.config.item_name_collection
        if not milvus_client.has_collection(item_name_collection):
            self._create_item_collection(item_name_collection,milvus_client)
        #构建数据行
        item_name_data_row={
            "file_title":file_title,
            "item_name":item_name,
            "dense_vector":dense_vecs,
            "sparse_vector":sparse_vecs
        }
        #插入数据
        insert_result=milvus_client.insert(item_name_collection,item_name_data_row)
    def _create_item_collection(self, item_name_collection,milvus_client):
        """
        创建商品名集合
        Args:
            item_name_collection: 集合名字
            milvus_client: 客户端

        Returns:

        """
        #1创建约束
        schame=milvus_client.create_schema()
        schame.add_field(
        field_name="pk",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=64
        ).add_field(
        field_name="file_title",
        datatype=DataType.VARCHAR,
        max_length=65535
        ).add_field(
        field_name="item_name",
        datatype=DataType.VARCHAR,
        max_length=65535
        ).add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024
        ).add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        #2创建索引
        index_params=milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="L2",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        #3创建集合
        milvus_client.create_collection(
            collection_name=item_name_collection,
            schema=schame,
            index_params=index_params

        )
        self.logger.info(f"创建商品名集合成功")

    def _fill_item_name(self, state, item_name,chunks):
        """
        回填item_name到chunk和state
        Args:
            state:
            item_name:
            chunks:

        Returns:

        """
        #更新chunks,这里的chunks是state中的引用，所以可以直接修改，state会自动改
        for chunk in chunks:
            chunk["item_name"]=item_name
        #更新state中的item_name
        state["item_name"]=item_name



if __name__ == '__main__':
    import json
    import os
    from knowledge.processor.import_processor.base import setup_logging
    setup_logging()

    # 路径配置
    base_dir = r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir"
    chunk_json_path = os.path.join(base_dir, "chunks.json")
    output_json_path = os.path.join(base_dir, "chunks_name.json")

    # 1. 读取chunks.json
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"读取chunks数量: {len(chunks)}")

    # 2. 实例化节点，利用其LLM方法提取item_name
    node = ItemNameRecognitionNode()
    config = node.config

    # 3. 构建商品名识别上下文（直接从chunks的content字段构建，兼容chunks.json格式）
    file_title = chunks[0].get("file_title", "未知标题") if chunks else "未知标题"
    item_name_chunk_k = config.item_name_chunk_k
    final_context = []
    for index, chunk in enumerate(chunks[:item_name_chunk_k]):
        if not isinstance(chunk, dict):
            continue
        # 优先取content字段，兼容context字段
        text = chunk.get("content") or chunk.get("context") or ""
        final_context.append(f"{index}_{text}")
    item_name_context = "\n".join(final_context)

    # 4. 调用LLM提取商品名
    item_name = node._use_llm_to_recognize_item_name(item_name_context, file_title)
    print(f"LLM识别的商品名: {item_name}")

    # 5. 回填item_name到每个chunk，生成新的chunks列表
    for chunk in chunks:
        chunk["item_name"] = item_name

    # 6. 保存为chunks_name.json
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"已生成: {output_json_path}")
    print(f"首个chunk示例: item_name={chunks[0]['item_name']}, title={chunks[0]['title']}")

    # 7. 对item_name进行向量化嵌入
    print("开始向量化...")
    dense_vecs, sparse_vecs = node._embedding_item_name(item_name)
    if dense_vecs and sparse_vecs:
        print(f"向量化成功: dense维度={len(dense_vecs)}, sparse维度={len(sparse_vecs)}")
    else:
        print("向量化失败，跳过Milvus写入")
        exit(1)

    # 8. 写入Milvus（带详细诊断）
    print("开始写入Milvus...")
    try:
        milvus_client = StorageClients.get_milvus()
        print(f"Milvus连接成功, 集合名: {config.item_name_collection}")
        # 检查集合是否已存在及其schema
        if milvus_client.has_collection(config.item_name_collection):
            col_info = milvus_client.describe_collection(config.item_name_collection)
            print(f"集合已存在, schema: {col_info}")
        else:
            print("集合不存在，将自动创建")
        node._insert_milvus(file_title, item_name, dense_vecs, sparse_vecs)
        # 验证写入结果
        stats = milvus_client.get_collection_stats(config.item_name_collection)
        print(f"写入完成, 集合统计: {stats}")
    except Exception as e:
        print(f"Milvus写入失败: {e}")
