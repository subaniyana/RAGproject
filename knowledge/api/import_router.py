import os

import uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.corn.deps import get_upload_file_service
from knowledge.corn.paths import get_front_page_dir
from knowledge.service.upload_service import UploadService
from knowledge.shcemal.upload_schema import UploadResponse,TaskStatusResponse
from knowledge.utils.task_util import get_task_info


def create_app():
    """
    创建fastapi实例
    Returns:fastapi实例
    """
    #1.创建fastapi实例
    app=FastAPI(description="知识库导入应用",version="0.1.0")

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
    @app.post("/upload",response_model=UploadResponse)
    def upload_endpoint(file:UploadFile,
                        background_task:BackgroundTasks,
                        upload_service:UploadService=Depends(get_upload_file_service)):
        """
        处理文件上传
        Returns:
        """
        #1.将上传文件写入到本地临时目录，以及远程minio中
        task_id,import_file_path,file_dir=upload_service.process_upload_file(file)

        #2.运行整个导入图谱(节点多耗时，使用后台做任务)
        background_task.add_task(upload_service.run_import_graph,task_id,import_file_path,file_dir)

        #3.返回响应后是数据模型
        return UploadResponse(message=f"{file.filename}上传成功",task_id="")

    #查询上传任务状态的请求
    @app.get("/status/{text_id}",)
    def get_status_endpoint(task_id:str):
        """
        查询上传任务状态：前端轮询调用查询上传任务状态
        轮询：性能和实时性
        1.5s
        Returns:
        """
        task_info=get_task_info(task_id)

        return TaskStatusResponse(**task_info)


if __name__ == '__main__':
    # 创建fastapi实例
    # 启动的服务器地址
    #启动的服务器端口
    # 启动服务器
    uvicorn.run(app=create_app(), host='0.0.0.0', port=8000, log_level="info")

