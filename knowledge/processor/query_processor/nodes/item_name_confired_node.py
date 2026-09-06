import re,logging,json
from json import JSONDecodeError
from typing import Dict, List, Optional, Tuple, Any

from knowledge.processor.query_processor.base import  BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.base import logger
from knowledge.prompt.query_prompt import *
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import execute_hybrid_search_query,create_hybrid_search_requests
from knowledge.processor.query_processor.base import get_config
from knowledge.utils.mongo_history_util import get_recent_messages


class _ItemNameExtractNode:

    def extact_item_name(self, original_query, history_context)->Dict:
        """
        提取商品名
        Args:
            original_query: 原始用户查询信息
            history_context: 历史对话上下文

        Returns:

        """
        #设置默认llm结果
        llm_result={"item_names": [],"rewritten_query":original_query}
        #获得llm客户端
        try:
            llm_client = AIClients.get_llm_openai()
        except ConnectionError as e:
            logger.error(f"获取llm客户端失败:{e}")
        #获得商品名提示词
        system_prompt=ITEM_NAME_EXTRACT_SYSTEM_PROMPT
        item_name_user_prompt=ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_context.strip() if history_context else "",
                                                                query=original_query)
        #调用llm
        try:
            llm_response=llm_client.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item_name_user_prompt}
            ])
        except Exception as e:
            logger.error(f"调用llm模型失败:{e}")
        llm_response_content=llm_response.content
        if not llm_response_content:
            return llm_result

        #清洗（判断输出内容的内型和空格）解析（反序列化）
        parsed_content:Dict[str,Any]=self._clean_and_parse(llm_response_content)

        #返回结果
        llm_result["item_names"]=parsed_content.get("item_names")
        llm_result["rewritten_query"]=parsed_content.get("rewritten_query") if parsed_content.get("rewritten_query") else original_query

        return llm_result

    def _clean_and_parse(self, llm_response_content)->Dict[str,Any]:
        """
        清洗（判断输出内容的内型和空格）解析（反序列化）
        Args:
            llm_response_content:

        Returns:

        """
        #去除json代码块围栏标记```{}```防止llm模型更换或者底层api调用升级使模型输出json报错
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        #解析
        try:
            # 反序列化
            llm_content_obj=json.loads(content)
            #获得item_names判断类型以及清洗
            raw_item_names=llm_content_obj.get("item_names")
            # 判断类型
            if not isinstance(raw_item_names,list):
                item_names=[]
            else:
                item_names=[item_name.strip() for item_name in raw_item_names if item_name.strip() and isinstance(item_name,str)]
            #获得rewritten_query
            rewritten_query=llm_content_obj.get("rewritten_query")
            if not isinstance(rewritten_query,str):
                rewritten_query=""
            else:
                rewritten_query=rewritten_query.strip()

            return {"item_names":item_names,"rewritten_query":rewritten_query}
        except JSONDecodeError as e:
            logger.error(f"json解析失败:{e},得到结果为{llm_response_content}")
            raise JSONDecodeError(msg=e.msg,doc=e.doc,pos=e.pos)


class _ItemNameAlinger:
    def __init__(self):
        self._config=get_config()

    def search_and_align(self, item_names:List[str])->Tuple[List[str],List[str]]:
        """
        检索向量数据库并和向量数据库中的商品名对齐，返回确定的商品名列表和模糊的商品名列表
        Args:
            item_names:llm提到的商品名列表

        Returns:confirmed,options
        """
        #混合检索向量数据库
        search_result:List[Dict[str,Any]]=self._search_vector(item_names)
        if not search_result:
            return [],[]
        #根据混合向量检索到的结果对齐
        confirmed,options=self._align(search_result)
        #返回确定的confired容器和模糊的options容器
        if len(confirmed)>1:
            confirmed=self._item_name_num_ensure(confirmed,search_result)

        return confirmed,options


    def _search_vector(self, item_names)->List[Dict[str,Any]]:
        """
        对llm提取到底所有桑排名进行向量检索
        Args:
            item_names:

        Returns:List[Dict[str,Any]]
            Dict[str,Any]:
            {"extracted_name":"商品名1"，"matches":[{向量库中文档1(score,item_name)}{向量库中文档2}]}
        """
        final_search_result=[]
        #获得milvus客户端,bge_m3模型
        try:
            milvus_client=StorageClients.get_milvus()
        except ConnectionError as e:
            logger.error(f"获取milvus客户端失败:{e}")
            return final_search_result
        try:
            bge_m3_model = AIClients.get_bge_m3()
        except ConnectionError as e:
            logger.error(f"获取bge_m3模型失败:{e}")
            return final_search_result

        #商品名向量化
        try:
            hybrid_vector_result=generate_bge_m3_hybrid_vectors(bge_m3_model,item_names)
        except Exception as e:
            logger.error(f"商品名{item_names}向量化失败:{e}")

        #混合向量检索
        #构建混合检索请求
        for index,item_name in enumerate(item_names):
            hybrid_requests=create_hybrid_search_requests(hybrid_vector_result["dense"][ index],
                                          hybrid_vector_result["sparse"][ index])
        #执行混合向量检索
            hybrid_search_result=execute_hybrid_search_query(milvus_client,
                                                             self._config.item_name_collection,
                                                             hybrid_requests,
                                                             output_fields=["item_name"])
        #解析结果并封装
            #获得检索结果得分与返回的商品名
            try:
                matches=[{"score":item["distance"],"item_name":item["entity"]["item_name"]} for item in (hybrid_search_result[0]if hybrid_search_result else [])]
                final_search_result.append({
                    "extracted_name":item_name,
                    "matches":matches
                })
            except Exception as e:
                logger.error(f"解析混合向量{hybrid_search_result}检索结果失败:{e}")
        return final_search_result

    def _align(self,search_result:List[Dict[str,Any]])->Tuple[List[str],List[str]]:
        """
        对齐规则
        1.如果向量数据库查询的商品名得分x>0.7放入confired
        2.如果向量数据库查询的商品名得分0.45<x<0.7放入options
        3.如果向量数据库查询的商品名得分x<0.45,都不放入
        阈值如何得到：

        Args:
            search_result: 向量数据库检索结果
            [{"extracted_name":"商品名1"，"matches":[{向量库中文档1}{向量库中文档2}]},]

        Returns:congired：一个item_name只取1个确定值，一组有多个确定值，不需要两个一模一样的商品名（给下游检索不需要两个一模一样的名字）
                options：一个item_name只取3个模糊值，一组也只取3个，不需要两个一模一样的商品名（不能给用户展示一模一样商品名）

        """
        confired=[]
        options=[]
        #遍历向量数据库查询结果
        for item in search_result:
            item_extract_name=item.get("extracted_name")
            item_matches=item.get("matches")
            #搜集score>0.7的结果
            high=[match for match in item_matches if match.get("score")>self._config.item_name_high_confidence]
            #如果高置信度结果存在
            if high:
                #1.判断是否有item_name完全匹配的
                extract=next((h for h in high if h.get("item_name")==item_extract_name),None)
                if extract:#完全匹配
                    picked=extract.get("item_name")
                    if picked not in confired:#去重
                        confired.append(picked)
                elif len(high)==1:#不完全匹配但唯一结果
                    piched=high[0].get("item_name")
                    if picked not in confired:
                        confired.append(piched)
                else:#多个结果
                    top_score=high[0].get("score")
                    if top_score-high[1].get("score")>self._config.item_name_score_gap:
                        picked=high[0].get("item_name")
                        if picked not in confired:
                            confired.append(picked)
                    else:
                        for h in high[:self._config.item_name_max_options]:
                            picked=h.get("item_name")
                            if picked not in options and picked not in confired:
                                options.append(picked)
            else:
                mid=[match for match in item_matches
                     if match.get("score")>self._config.item_name_mid_confidence
                     and match.get("item_name") not in options
                     and match.get("item_name") not in confired]
                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        options.append(m.get("item_name"))
            #搜集score>0.45且score<0.7的结果
            mid=[match for match in item_matches if self._config.item_name_high_confidence>match.get("score")>self._config.item_name_score_gap]
            if mid:
                options.append(item_extract_name)
        return confired,options[:self._config.item_name_max_options]

    def _item_name_num_ensure(self, confirmed:List[str], search_results:List[Dict[str,Any]]):
        """
        确保llm没有一开始就识别商品重复出错（例如：rm收音机多少钱？提取为item_name:[rm收音机，收音机]）
         0.95 vs 0.78 这种巨大落差时，说明其中一个词（0.78 的那个）在库里根本没有精准匹配项，
         它只是碰瓷了一个相似的商品。这种词，大概率就是 LLM 把“品类词”错误提取成了“商品名”。
        Args:
            confirmed:确认商品名容器
            search_result:List[Dict[str,Any]

        Returns:

        """
        # 1. 构建 商品名 → 最高分数 的映射
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get('matches', [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        # 2. 防御性检查：如果没有收集到任何分数，直接返回原始 confirmed
        if not item_name_score:
            return confirmed

        # 3. 取出分数值最大的作为基准
        max_score = max(item_name_score.values())
        return [name for name, score in item_name_score.items()
                if max_score - score <= self._config.item_name_score_gap]


class ItemNameConfiredNode(BaseNode):
    name = "item_name_confired_node"
    def __init__(self):
        super().__init__()
        self._extractor= _ItemNameExtractNode()
        self._aligner = _ItemNameAlinger()
    def process(self, state:QueryGraphState):
        """
        1。利用llm从用户原始查询中提取到商品名并改写原始查询
            1.1如果llm提取到商品名，则进行第2步与milvus对齐
            1.2如果llm没有提取到商品名，直接返回
        2.根据milvus中存储的商品名进行对齐（使检索更加准确，三路检索利用到节点提取到的商品名，如果直接用llm提取的商品名在三路检索过滤时，过滤条件不准确
            检索噪音过多，llm输出幻觉大，最终根据milvus表格存储的商品名进行检索,因为这样每一个chunk都对应一个商品名）
        3.决策（该往下走还是回头）


        两个容器：（利用两个容器产生三个分支）
        1.confired:如果精确商品名，给confired添加精确商品名
        2.optins:商品名不精确没找到多个相似的，给options中添加找到的多个不确定商品名

        Returns:
            state["answer"]不给，进行三路检索；获得三路检索结果给llm，llm生成答案给state["answer"]
            state["answer"]
            1.返回候选商品名给用户候选使用
            2.没有任何商品名，则返回抱歉，没有找到商品名

        """

        #获得用户原始问题
        original_query = state.get("original_query")
        session_id=state.get("session_id")
        #获得历史对话
        history_context=get_recent_messages(session_id,10)#获得最近10条信息，5轮对话

        format_history=[]
        role_map = {"user": "用户", "assistant": "助手"}
        for history in history_context:
            role = history.get('role', '')
            text = history.get('text', '')

            # 1.3 获取格式化后的行
            if not text or role not in role_map:
                continue

            formatted_line = f"角色：{role_map[role]}: 内容：{text}"
            format_history.append(formatted_line)
        format_history_str = " ".join(format_history)


        #利用llm进行商品名提取与查询重写
        llm_result:Dict[str,Any]=self._extractor.extact_item_name(original_query,format_history_str )

        #获得llm返回结果
        item_names=llm_result.get("item_names")
        rewritten_query=llm_result.get("rewritten_query")
        if item_names:
            confired, options = self._aligner.search_and_align(item_names, rewritten_query)
        else:
            confired, options = [],[]

        #决策
        self._deside(confired, options,state,item_names,rewritten_query)
        state["history"]=format_history
        return state

    def _deside(self, confired, options, state, item_names, rewritten_query):
        """
        根据confired还是options来判断是否继续检索还是返回用户提升信息
        Args:
            confired:已经确认的商品名列表
            options:模糊的商品名列表
            state:查询状态
            item_names:商品名列表
            rewritten_query:重写后的问题

        Returns:
        """
        if confired:
            state["item_names"]=confired
            state["rewritten_query"]=rewritten_query
        elif options:
            #  有候选：设置 answer 为反问文案
            # state 中有 answer → main_graph 条件边走 "直接返回"
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            # 无匹配：设置 answer 为无法识别文案
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"





if __name__ == '__main__':
    from knowledge.processor.import_processor.base import setup_logging
    setup_logging(logging.INFO)  # 配置日志
    item_name_confired_node=ItemNameConfiredNode()
    init_state={
        "original_query": "怎么用RS-12万用表测电阻，华为云计算DDH-99怎么测量",
    }
    llm_result=item_name_confired_node.process(init_state)
    print(llm_result)

