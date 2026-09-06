
from typing import Dict, List, Tuple, Any, Optional


from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.base import BaseNode, T


class RRFMergeNode(BaseNode):
    name="rrf_merge_node"
    def process(self, state:QueryGraphState) -> QueryGraphState:
        #获得本地检索的两类结果
        enbedding_chunks=state.get("embedding_chunks") or  []
        hybe_embedding_chunks=state.get("hyde_embedding_chunks") or []
        #定义两路检索结果和对应路的权重映射表
        search_results_weights={
            "enbedding_chunks":(self._validate(enbedding_chunks),1.0),
            "hybe_embedding_chunks":(self._validate(hybe_embedding_chunks),1.0)
        }
        rrf_inputs=list(search_results_weights.values())
        #利用RRF计算两路文档分数,去重并降序排序
        rrf_merge_result:List[Tuple[Dict[str, Any], float]]=self._merge_rrf_docs(rrf_inputs,self.config.rrf_k,self.config.rrf_max_results)

        state["rrf_chunks"]=rrf_merge_result
        return state

    def _merge_rrf_docs(self, rrf_inputs:List[Tuple[Dict[str, Any], float]], rrf_k:int, rrf_max_results:int)->List[Tuple[Dict[str, Any], float]]:
        """
        利用RRF计算多路检索返回的文档分数
        公式：weight(i)/[k+rank(i)]
        Args:
            rrf_inputs: 两路稳当输入
            rrf_k: 平滑参数常数k的值
            rrf_max_results: 返回文档数目

        Returns:
            返回多路检索后文档对象与得分,排序从大到小降序
            Tuple[Dict,float]前为文档对象只有entity中的内容，后为得分
        """
        #id到entity对象的映射
        id_entity_map={}
        #存储每个文档得分
        rrf_scores={}

        #遍历计算每个路径检索结果
        for search_result,weight in rrf_inputs:
            #遍历某一路径的文档并计算得分
            for i,entity in enumerate(search_result):
                chunk_id=entity.get("chunk_id")
                if not chunk_id:
                    continue
                rrf_scores[chunk_id]=rrf_scores.get(chunk_id,0.0)+weight/(rrf_k+i)
                #建立id到entity的映射，且只会在key没有时才执行
                id_entity_map.setdefault(chunk_id, entity)
        #排序并构建chunk对象和得分结果
        final_rrf_result=sorted([(id_entity_map.get(chunk_id),score) for chunk_id,score in rrf_scores.items()],
                                key=lambda x:x[1],reverse=True)
        return final_rrf_result[:rrf_max_results] if rrf_max_results else final_rrf_result


    def _validate(self, enbedding_chunks:List[Dict[str, Any]])->List[Dict[str, Any]]:
        """
        校验检索结果
        Args:
            enbedding_chunks:

        Returns:只返回entity中内容（List[Dict[str, Any]）

        """
        entity_results=[]
        if not enbedding_chunks:
            return []
        for chunk in enbedding_chunks:
            if not chunk or not isinstance(chunk,dict):
                continue
            entity=chunk.get("entity")
            if not entity or not isinstance(entity,dict):
                continue
            entity_results.append(entity)
        return entity_results

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RRFMergeNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk[0].get('chunk_id')} - {chunk[0].get('content')}")

    print("-" * 60)
    print("测试完成")