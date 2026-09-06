from typing import Dict, List, Tuple, Any, Union

from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.utils.client.ai_clients import AIClients
from knowledge.processor.query_processor.exceptions import *
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, _item_names_filter, execute_hybrid_search_query


class HybridVectorSearchNode(BaseNode):
    name="hybrid_vector_search_node"

    def process(self, state:QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:
        #参数校验
        user_question,item_names=self._check_params(state)

        #获得嵌入模型客户端
        try:
            bge_m3_client=AIClients.get_bge_m3()
        except Exception as e:
            self.logger.error(f"获得bge-m3嵌入模型客户端失败:{e}")
            return state

        #获得milvus客户端
        try:
            milvus_client=StorageClients.get_milvus()
        except Exception as e:
            self.logger.error(f"获取milvus客户端失败:{e}")
            return state

        #嵌入查询问题
        try:
            embed_query_vector=generate_bge_m3_hybrid_vectors(bge_m3_client,[user_question])
        except:
            self.logger.error(f"用户问题{user_question}嵌入失败")
            return state
        #检索问题
        try:
            # 过滤器
            expr, expr_params = _item_names_filter(item_names)
            # 创建milvus查询请求
            hybrid_search_req=create_hybrid_search_requests(dense_vector=embed_query_vector.get("dense")[0],
                                          sparse_vector=embed_query_vector.get("sparse")[0],
                                          expr=expr,
                                          expr_params=expr_params,
                                          limit=5)
            #执行milvus查询请求
            hybrid_search_result=execute_hybrid_search_query(milvus_client,
                                        collection_name=self.config.chunks_collection,
                                        search_requests=hybrid_search_req,
                                        output_fields=["chunk_id","content","item_name","title"])
            if not hybrid_search_req or not hybrid_search_result[0]:
                return state

        except Exception as e:
            self.logger.error(f"用户问题{user_question}查询向量失败:{e}")

            return {"embedding_chunks":hybrid_search_result[0]}

    def _check_params(self, state:QueryGraphState):
        """
        返回用户问题与商品名列表
        """
        #用户问题与商品名称
        user_question=state.get("rewritten_query")
        item_names=state.get("item_names")

        if not user_question or not isinstance(user_question,str):
            raise StateFieldError(node_name=self.name,field_name="rewritten_query")
        if not item_names or not isinstance(item_names,list):
            raise StateFieldError(node_name=self.name,field_name="item_names")
        return user_question,item_names

if __name__ == '__main__':
    import json

    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    vector_search = HybridVectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))

