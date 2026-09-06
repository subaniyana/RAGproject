import asyncio
import os
from typing import Union

import uvicorn
from fastapi import FastAPI,  Depends, BackgroundTasks,Request,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from knowledge.corn.deps import get_query_service
from knowledge.corn.paths import get_front_page_dir
from knowledge.service.query_service import QueryService
from knowledge.shcemal.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.utils.sse_util import create_sse_queue,sse_generator
from knowledge.utils.task_util import get_task_info,get_task_result



def create_app():
    """
    创建fastapi实例
    Returns:fastapi实例
    """
    #1.创建fastapi实例
    app=FastAPI(description="知识查询应用",version="0.1.0")

    #2.跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # ← 默认值，可以省略
        allow_methods=["*"],
        allow_headers=["*"],
    )
    #3.挂载静态文件
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
    # 注册路由
    register_router(app)
    return app

def register_router(app:FastAPI):
    """
    注册路由
    """
    #1.上传请求
    @app.post("/query",response_model=Union[StreamSubmitResponse,QueryResponse])
    async def query(request:QueryRequest,
              background_task: BackgroundTasks,
              service:QueryService=Depends(get_query_service))->Union[StreamSubmitResponse,QueryResponse]:
        """
        运行graph图谱，返回task_id
        处理查询请求
        Args:
            request:
            background_task:
            service:

        Returns:

        """
        #获得session_id
        session_id=request.session_id or service.generate_session_id()

        #获得task_id
        task_id=service.generate_task_id(session_id)

        #如果是流式调用
        if request.is_stream:
            #创建sse队列
            create_sse_queue(task_id)
            #后台运行graph
            background_task.add_task(service.run_query_graph,task_id,session_id,request.query,request.is_stream)
            #返回响应
            return StreamSubmitResponse(
                message="查询已提交",session_id=session_id,task_id=task_id
            )

        #非流式调用
        else:
            loop=asyncio.get_event_loop()
            await loop.run_in_executor(None,service.run_query_graph,task_id,session_id,request.query,request.is_stream)
            #运行查询功能graph
            service.run_query_graph(task_id,session_id,request.query,request.is_stream)
            #graph结果会存储在task队列中，从中取出answer
            task_answer_str=get_task_result(task_id,key="answer")
            #返回响应
            return QueryResponse( message="查询已提交",session_id=session_id,answer=task_answer_str)



    #查询上传任务状态的请求
    @app.get("/stream/{text_id}")
    async def stream(task_id:str,request:Request):
        """
        返回sse协议要的数据包
        1.利用生成器yeild返回
        2.注意事项：event:自定义\ndata:自定义\n\n
        Returns:
        """
        return StreamingResponse(content=sse_generator(task_id,request),media_type="text/enent-stream")

    @app.get("history/{session_id}")
    async def get_history(
            session_id:str,
            limit:int=30,
            service:QueryService=Depends(get_query_service)
    ):
        try:
            items=service.get_history(session_id,limit)
            return {"session_id":session_id,"items":items}
        except Exception as e:
            raise HTTPException(status_code=500,detail=str(e))

    @app.delete("history/{session_id}")
    async def delete_history(
            session_id:str,
            service:QueryService=Depends(get_query_service)
    ):
        try:
            count=service.delete_history(session_id)
            return {"message":"删除成功","deleted_count":count}
        except Exception as e:
            raise HTTPException(status_code=500,detail=str(e))

if __name__ == '__main__':
    # 创建fastapi实例
    # 启动的服务器地址
    #启动的服务器端口
    # 启动服务器
    uvicorn.run(app=create_app(), host='0.0.0.0', port=8000, log_level="info")

