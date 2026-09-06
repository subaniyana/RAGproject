import logging
import math
from typing import Dict, List, Tuple, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.utils.client.ai_clients import AIClients
from knowledge.processor.query_processor.exceptions import *
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, _item_names_filter, execute_hybrid_search_query
from knowledge.prompt.hybe_prompt import *
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import SSEEvent, push_sse_event
from knowledge.utils.task_util import get_task_result,set_task_result
from knowledge.prompt.query_prompt import ANSWER_PROMPT

class AnswerOutputNode(BaseNode):
    name="answer_output_node"
    def process(self, state:QueryGraphState) -> QueryGraphState:
        """
        从state中获得answer输出答案
        1.如果获得到answer--->没有三路检索（商品名识别不成功的，不用llm,直接询问用户）【流式推送SSE/非流式推送】
        2.如果获得到answer----->进行了三路检索（商品名识别成功的，调用llm）【流式推送SSE/非流式推送】
        Args:
            state:

        Returns:
        """
        #是否流式生成
        is_stream=state.get("is_stream")
        #获得任务id
        task_id=state.get("task_id")

        #判断是否有answer如果有answer则直接推送出去
        if state.get("answer"):
            self._push_exist_answer(state,is_stream,task_id)
            is_streamed=False

        #如果没有answer则结合三路检索内容交给llm生成answer
        else:
            #组装提示词
            prompt=self._build_prompt(state)
            state["prompt"]= prompt
            is_streamed=True

            #调用llm
            self._generate_answer(state,prompt,is_stream,task_id)

        #保存历史对话
        self.save_history(state)

        #告诉前端sse通道关闭
        if is_stream:
            #已经使用过流式生成答案
            if is_streamed:
                push_sse_event(task_id, SSEEvent.FINAL, {})
            #没有使用流式生成答案，当没有使用三路检索直接推送答案
            else:
                push_sse_event(task_id, SSEEvent.FINAL, {"answer": state.get("answer")})

        return state


    def _push_exist_answer(self, state:QueryGraphState, is_stream:bool, task_id:str):
        """
        推送
        Args:
            state:
            is_stream:
            task_id:

        Returns:

        """
        #判断为非流式队列
        if not is_stream:
            set_task_result(task_id, "answer",value=state.get("answer"))


    def _build_prompt(self, state:QueryGraphState)->str:
        #获得基本参数
        max_context_chars = self.config.max_context_chars
        user_question=state.get("rewritten_query")
        item_names=state.get("item_names") or []
        
        #获得检索到的上下文
        searched_contents=state.get("reranked_docs") or []
        format_content,_=self._format_content(searched_contents,max_context_chars)

        #获得历史对话
        chat_history=state.get("history") or []#已经从mongo中获得了对话放进内存中
        history=self._format_history(chat_history)

        #组装提示词
        prompt=ANSWER_PROMPT.format(
            context=format_content or "",
            history=history or "",
            item_names=",".join(item_names),
            question=user_question,
        )
        return prompt



    def _format_content(self, searched_contents:List,max_context_chars:int)->Tuple[str,int]:
        """
        格式化检索到的上下文
        Args:
            searched_content:
            max_context_chars:

        Returns:

        """

        formatted_lines = []
        used_chars = 0
        # 1. 遍历格式化后的文档
        for index, doc in enumerate(searched_contents):

            # 1.1 获取内容
            content = doc.get('content', '')

            # 1.2 判断内容是否存在
            if not content:
                continue

            # 1.3 构建元数据
            content_meta = [f"[{index}]"]

            # 1.3.1 拼接元数据（模板统一为 [字段名=值]，仅拼接非空字段）
            for meta_field in ("source", "chunk_id", "url", "title"):
                # a. 获取元数据字段值
                meta_field_value = str(doc.get(meta_field) or '').strip()

                # b. 填充模版变量
                if meta_field_value:
                    content_meta.append(f"[{meta_field}={meta_field_value}]")

            # 1.4 获取分数
            doc_score = doc.get('score')
            if doc_score is not None:
                content_meta.append(f"[score={float(doc_score):.4f}]")

            # 1.5 构建完整文档
            doc_entry = " ".join(content_meta) + "\n" + content

            # 1.6 分隔符长度“\n\n”最开始的一段不需要
            separator_usage = 2 if formatted_lines else 0

            # 1.7 计算总长度
            total_usage = separator_usage + len(doc_entry)
            if used_chars + total_usage > max_context_chars:
                break

            # 1.8 将当前行加入
            formatted_lines.append(doc_entry)
            used_chars += total_usage

        return "\n\n".join(formatted_lines), max_context_chars - used_chars




    def _generate_answer(self, state:QueryGraphState, prompt:str, is_stream:bool, task_id:str):
        """
        调用llm生成答案更新到state
        Args:
            state:
            prompt:
            is_stream:
            task_id:

        Returns:

        """
        #获得客户端
        try:
            llm_client=AIClients.get_llm_openai(resonse_format=False)
        except Exception as e:
            self.logger.error(f"获得llm_client失败:{e}")
            state["answer"]="llm客户端获取失败，无任何回答"
            return

        if is_stream:
            #流式调用llm
            state["answer"]=self._stream_invoke_llm(prompt,llm_client,task_id)
        else:
            #非流式调用llm
            state["answer"]=self._invoke_llm(prompt,llm_client)
            #写入非流式调用队列中
            set_task_result(task_id, "answer",value=state["answer"])

    def _invoke_llm(self, prompt:str, llm_client:ChatOpenAI)->str:
        """
        非流式llm调用
        Args:
            prompt:
            llm_client:

        Returns:
        """
        llm_result=llm_client.invoke(prompt)
        llm_content=getattr(llm_result,"content","") or ""

        if not llm_content:
            return "llm无回答"
        return llm_content

    def _stream_invoke_llm(self, prompt, llm_client, task_id)->str:
        """
        流式调用llm
        Args:
            prompt:
            llm_client:
            task_id:

        Returns:
        """
        #获得全量数据
        accelerate_delta=""
        try:
            #流式调用
            for chunk in llm_client.stream(prompt):
                delta=getattr(chunk,"content","") or ""
                if delta:#获得增量数据
                    push_sse_event(task_id, SSEEvent.DELTA, {"answer": delta})
                    accelerate_delta+=delta
        except Exception as e:
            return "llm无回答"
        return accelerate_delta

    def save_history(self, state:QueryGraphState):
        """
        保存历史对话在mongoDB对应kb001库的表chat_message表中
        Args:
            state:

        Returns:
        """
        #1. 获得会话ID
        session_id=state.get("session_id")

        #2. 获得原始查询信息
        user_qury=state.get("original_query")

        #3. 获得修改后的查询信息
        rewritten_query=state.get("rewritten_query")

        #4.获得商品名
        item_names=state.get("item_names") or []

        if state.get("answer"):
            try:
                #保存用户消息
                save_chat_message(session_id, "user", user_qury,rewritten_query,item_names)
                #保存ai回复消息
                save_chat_message(session_id, "assistant", state.get("answer"),rewritten_query,item_names)
            except Exception as e:
                self.logger.error(f"mongo保存历史对话失败:{e}")
            pass

    def _format_history(self, chat_history:List[Dict[str,str]], char_budget:int=2000)->str:
        """
        格式化上下文
        Args:
            chat_history:

        Returns:

        """
        formatted_lines = []
        used_chars = 0
        # 1. 遍历格式化后的文档
        role_map = {"user": "用户", "assistant": "助手"}
        for msg in chat_history:
            # 1.1 获取消息角色
            role = msg.get('role', '')

            # 1.2 获取消息内容
            text = msg.get('text', '')

            # 1.3 获取格式化后的行
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"

            # 1.4 计算分割符长度
            seperator_usage = 1 if formatted_lines else 0

            # 1.5 计算总长度
            total_usage = seperator_usage + len(formatted_line)

            if used_chars + total_usage > char_budget:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return "\n".join(formatted_lines)


