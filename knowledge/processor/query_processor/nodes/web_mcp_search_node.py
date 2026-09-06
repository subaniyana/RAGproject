import asyncio
import json
from json import JSONDecodeError
from typing import Dict, List, Tuple, Any, Optional,Union


from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.exceptions import *
from agents.mcp import MCPServerStreamableHttp


class WebMcpSearchNode(BaseNode):
    name="web_mcp_search_node"
    def process(self, state:QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:
        #参数校验
        rewritten_query, item_names = self._check_params(state)
        #定义并执行mcp调用
        #调用异步方法要么在异步环境中使用await,要么在同步环境中使用异步环境
        web_search_mcp_result :List[Dict[str,str]]= asyncio.run(self._execute_mcp(rewritten_query))
        if not web_search_mcp_result:
            return state
        #更新state
        return {"web_search_docs":web_search_mcp_result}



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

    async def _execute_mcp(self, rewritten_query:str):
        """
        执行mcp服务
        一个mcp服务下有多个工具
        Args:
            user_question:

        Returns:

        """
        #1.定义mcp客户端（streamablehttp）
        async with MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": self.config.mcp_dashscope_base_url,  # MCP 服务端点
                "headers": {"Authorization":  f"Bearer {self.config.mcp_dashscope_api_key}"},  # 认证头
                "timeout": 10,  # 请求超时时间（秒）
                "terminate_on_close": True,  # 关闭时终止连接
            },
            max_retry_attempts=2,  # 最大重试次数
            cache_tools_list=True,  # 缓存工具列表，避免重复请求
        )as mcp_client:

            # 建立连接（客户端和服务端）,交给上下文管理器后，自动连接
            # await mcp_client.connect()

            #2.调用工具
            try:
                web_search_result=await mcp_client.call_tool(tool_name="bailian_web_search",arguments={"query":rewritten_query,"count":2})
            except Exception as e:
                self.logger.error(f"调用工具失败:{e}")

            # 上下文管理器自动关闭连接
            # await mcp_client.cleanup()
            # 3.解析数据
            #3.1获得文本内容块
            json_content=web_search_result.content[0]
            if not json_content:
                return []
            #3.2获得文本内容快对象内容
            json_text=json_content.text
            if not json_text:
                return []
            try:
                # 反序列化json数据
                text_dict=json.loads(json_text)
                #获得网页内容
                pages=text_dict.get("page",[])
                if not pages:
                    return []
                #遍历
                web_search_result=[]
                for page in pages:
                    web_search_result.append({
                        "title":page.get("title").strip(),
                        "url":page.get("url").strip(),
                        "snippet":page.get("snippet").strip()
                    })
                return web_search_result
            except JSONDecodeError as e:
                self.logger.error(f"反序列化json数据失败:{e},失败信息{e.msg},失败内容{e.doc},失败位置{e.pos}")
                return []



if __name__ == '__main__':
    web_mcp_search_node=WebMcpSearchNode()
    state={
        "rewritten_query":"aespa女团成员有谁",
        "item_names":["python"]
    }
    web_mcp_search_node.process(state)