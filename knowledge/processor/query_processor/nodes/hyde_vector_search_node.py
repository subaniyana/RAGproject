from typing import Dict, List, Tuple, Any, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.utils.client.ai_clients import AIClients
from knowledge.processor.query_processor.exceptions import *
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, _item_names_filter, execute_hybrid_search_query
from knowledge.prompt.hybe_prompt import *

class HyDeVectorSearchNode(BaseNode):
    name="hyde_vector_search_node"

    def process(self, state:QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:
        #参数校验
        user_question, item_names=self._check_params(state)
        #利用llm生成原始问题对应的假设性答案（解决跨域不对称）
        hyde_document=self._generate_hypothesis_answer(user_question, item_names)

        if hyde_document is None:
            return state

        # 获得嵌入模型客户端
        try:
            bge_m3_client = AIClients.get_bge_m3()
        except Exception as e:
            self.logger.error(f"获得bge-m3嵌入模型客户端失败:{e}")
            return state

        # 获得milvus客户端
        try:
            milvus_client = StorageClients.get_milvus()
        except Exception as e:
            self.logger.error(f"获取milvus客户端失败:{e}")
            return state

        # 嵌入假设性文档
        try:
            hyde_vector = generate_bge_m3_hybrid_vectors(bge_m3_client, [hyde_document])
        except:
            self.logger.error(f"假设性文档{hyde_vector}嵌入失败")
            return state
        # 检索问题
        try:
            # 过滤器
            expr, expr_params = _item_names_filter(item_names)
            # 创建milvus查询请求
            hybrid_search_req = create_hybrid_search_requests(dense_vector=hyde_vector.get("dense")[0],
                                                              sparse_vector=hyde_vector.get("sparse")[0],
                                                              expr=expr,
                                                              expr_params=expr_params,
                                                              limit=5)
            # 执行milvus查询请求
            hybrid_search_result = execute_hybrid_search_query(milvus_client,
                                                               collection_name=self.config.chunks_collection,
                                                               search_requests=hybrid_search_req,
                                                               output_fields=["chunk_id", "content", "item_name","title"])
            if not hybrid_search_req or not hybrid_search_result[0]:
                return state

        except Exception as e:
            self.logger.error(f"原始问题{user_question}的假设性文档{hyde_document}查询向量失败:{e}")

            return {"hyde_embedding_chunks":hybrid_search_result[0]}


    def _check_params(self, state):
        """
        返回用户问题str与商品名列表list
        """
        user_question = state.get("rewritten_query")
        item_names = state.get("item_names")

        if not user_question or not isinstance(user_question, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query")
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names")

        return user_question, item_names

    def _generate_hypothesis_answer(self, user_question, item_names)->Optional[str]:
        """
        生成假设性文档
        Args:
            user_question:
            item_names:
        Returns:llm_response.content.strip()：llm调用返回内容

        """
        try:
            llm=AIClients.get_llm_openai(resonse_format=False)
        except Exception as e:
            self.logger.error(f"获取llm客户端失败:{e}")
            return None
        #获得系统提示词
        system_prompt=SYSTEM_HYPE_PROMPT_TEMPLATE.format(validate_item_names=item_names)
        #获得用户提示词
        user_prompt=USER_HYDE_PROMPT_TEMPLATE.format(item_names=item_names,rewritten_query=user_question)
        #调用生成假设性答案
        try:
            llm_response=llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            self.logger.error(f"生成{item_names}的假设性答案出错:{e},")
            return None

        if not llm_response.content.strip():
            return None
        return llm_response.content.strip()
