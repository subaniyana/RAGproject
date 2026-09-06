import math
from typing import Dict, List, Tuple, Any, Optional


from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.utils.client.ai_clients import AIClients


class RerankNode(BaseNode):
    name="rerank_node"

    @staticmethod
    def _sigmoid(x):
        """sigmoid归一化"""
        return 1.0 / (1.0 + math.exp(-x))

    def process(self, state:QueryGraphState) -> QueryGraphState:
        """
        三重检索合并并使用reranker模型问答对重排序文档
        Args:
            state:

        Returns:

        """
        #1.获得用户问题
        user_question=state.get("rewritten_query")
        #2.获得两路检索结果rrh与web
        rerank_inputs:List[Dict[str,Any]]=self._rerank_input(state)
        #3.利用reranker进行精排
        refine_docs=self._refine_reranker(user_question,rerank_inputs)
        #4.动态调整返回文档条数k
        #在min_top_k3个到max_top_k10个之间，根据llm的输出结果动态调整返回文档条数k
        # k设置大了，llm上下文过长，截取到无关上下文，使llm产生幻觉
        # k设置小了，llm上下文过短，无法获取到有效信息，使llm产生幻觉
        reranked_docs=self._cliff_cutoff(refine_docs,self.config.rerank_min_top_k,self.config.rerank_max_top_k)
        state["reranked_docs"]=reranked_docs
        return state
    def _rerank_input(self, state:QueryGraphState):
        """
        获得两路检索结果
        Args:
            state:

        Returns:

        """
        all_doc_formated=[]
        #获得本地检索结果
        rrf_chunks=state.get("rrf_chunks") or []
        for chunk,_ in rrf_chunks:
            if not chunk or not isinstance(chunk,dict):
                continue
            #获得chunk中的content,title,chunk_id
            chunk_content=chunk.get("content",'')
            if not chunk_content:
                 continue
            chunk_title=chunk.get("title",'')
            chunk_id=chunk.get("chunk_id")
            #格式化文档
            format_result=self._format_doc(content=chunk_content,chunk_id=chunk_id,title=chunk_title,source="local")
            all_doc_formated.append(format_result)

        #获得web检索结果
        web_chunks=state.get("web_search_docs") or []
        for wchunk in web_chunks:
            if not wchunk or not isinstance(wchunk,dict):
                continue
            # 获得chunk中的content,title,chunk_id
            wchunk_content=wchunk.get("snippet",'')
            wchunk_title=wchunk.get("title",'')
            wchunk_url=wchunk.get("url",'')
            #格式化文档
            format_result=self._format_doc(content=wchunk_content,title=wchunk_title,source="web",url=wchunk_url)
            all_doc_formated.append(format_result)
        self.logger.info(f"所有文档格式化结果个数为:{len(all_doc_formated)}")
        return all_doc_formated




    def _format_doc(self, content:str, chunk_id:int=None, title:str="", source:str="", url:str=""):
        """
        格式化本地和web检索到的文档
        Args:
            content:文档内容
            chunk_id:检索到的文档标号
            title:标题
            source:来源web or local
            url:

        Returns:dict{
            "content":content,
            "chunk_id":chunk_id,
            "title":title,
            "source":source,
            "url":url
        }
        """
        return{
            "content":content,
            "chunk_id":chunk_id,
            "title":title,
            "source":source,
            "url":url
        }


    def _refine_reranker(self, user_question:str, rerank_inputs:List[Dict[str,Any]]):
        """
        使用reranker模型对rrf排序后的doc与web的doc进行打分并排序
        Args:
            user_question: 用户查询
            rerank_inputs: 本地和远程检索后的结果

        Returns:

        """
        #获得reranker模型
        try:
            reranker_model=AIClients.get_bge_reranker()
        except ConnectionError as e:
            self.logger.error(f"获取reranker模型失败:{e}")
            return [{**d,"score":None} for d in rerank_inputs]
        #构建Q->D的输入对
        query_and_docs=[(user_question,doc.get("content")) for doc in rerank_inputs]
        try:
            #计算打分（reranker得分范围很大且有正有负）
            reranker_scores=reranker_model.compute_score(sentence_pairs=query_and_docs)

            #组合最终结果
            doc_scores=[{**d, "score": self._sigmoid(float( score))} for d, score in zip(rerank_inputs, reranker_scores)]
            #排序
            sorted_docs=sorted(doc_scores,key=lambda x:x["score"],reverse=True)
            return sorted_docs

        except Exception as e:
            self.logger.error(f"使用reranker模型对的docs进行打分并排序失败:{e}")
            return [{**d, "score": None} for d in rerank_inputs]

    def _cliff_cutoff(self, refine_docs:List[Dict[str,Any]], rerank_min_top_k:int, rerank_max_top_k:int)->List[Dict[str,Any]]:
        """
        动态调整返回文档条数k
        初始为max返回，但是通过遍历文档的分数，找到分数断崖检查截断
        相对阈值：根据文档分数比例来截断
        绝对阈值：差值超过就视为断崖
        Args:
            refine_docs: 精排后的文档
            rerank_min_top_k: 最小返回文档条数
            rerank_max_top_k: 最大返回文档条数

        Returns:
        """

        #定义两个索引
        upper_bound=min(rerank_max_top_k,len(refine_docs))
        lower_bound=min(rerank_min_top_k,upper_bound)

        #截断标记
        cut_off_index =upper_bound
        max_gap=0

        for i in range(0,upper_bound-1):
            #获得当前文档和下一个文档的分数
            current_doc_score=refine_docs[i].get("score")
            next_doc_score=refine_docs[i+1].get("score")

            #跳过无分数的文档
            if not current_doc_score or not next_doc_score:
                continue

            gap=current_doc_score-next_doc_score

            if gap>=0.15 and gap>max_gap:#根据绝对阈值评估时
                max_gap=gap
                cut_off_index=i+1
                self.logger.info(f"当前文档的索引为:{i}发生了断崖,当前文档的分数为:{current_doc_score},下一个文档的分数为:{next_doc_score}")


        #不管在哪里断崖，至少保留lower_bound个文档
        cut_off_index=max(lower_bound,cut_off_index)
        cutoff_docs=refine_docs[:cut_off_index]
        return cutoff_docs

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    from knowledge.processor.import_processor.base import setup_logging

    setup_logging()  # 配置日志

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        # 与RRFMergeNode真实输出结构一致: List[Tuple[Dict, float]]
        "rrf_chunks": [
            ({"chunk_id": "local_1", "title": "主板维修手册",
              "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"}, 0.032),
            ({"chunk_id": "local_2", "title": "闲聊",
              "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"}, 0.016),
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")




