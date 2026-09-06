import logging
import shutil
import time
from datetime import datetime
import os
import uuid

from fastapi import UploadFile

from knowledge.corn.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, add_running_task, add_done_task, add_node_duration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UploadService:
    """
    处理相关文件上传的逻辑
    """
    def get_base_dir(self):
        """
        获得文件存储目录
        """
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))


    def run_import_graph(self, task_id:str, import_file_path:str, file_dir:str):
        """
        运行整个图片
        Args:
            task_id:
            import_file_path:
            file_dir:

        Returns:
        """
        #更新任务状态
        update_task_status(task_id,TASK_STATUS_PROCESSING)


        state = ImportGraphState(
            {
                "task_id":"",
                "import_file_path": r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用.pdf",
                "file_dir": r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir"
            }
        )
        try:
            for event in import_app.stream(state):
                for key, value in event.items():
                    logger.info(f"当前执行节点是-->{key}")
            update_task_status(task_id,TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"任务{task_id}输入过程中执行失败:{e}")
            update_task_status(task_id,TASK_STATUS_FAILED)

    def process_upload_file(self,file:UploadFile):
        """
        处理文件上传
        1.将上传的文件存储到本地临时目录
        2.将上传的文件存储到远程minio
        3.将task_id:str, import_file_path:str, file_dir:str返回
        Args:
            file:UploadFile
        Returns:task_id:str, import_file_path:str, file_dir:str
        """
        #生成任务id
        task_id=uuid.uuid4().hex[:8]
        add_running_task(task_id,"upload_file")
        start_time=time.time()

        #生成日期目录并拼接temp临时文件路径
        base_dir=self.get_base_dir()

        #构建文档完整目录
        file_dir=os.path.join(base_dir,task_id)

        #保存文件到临时目录
        import_file_path=self.save_file_to_local(file_dir,file)

        #保存文件到远程minio
        self.save_file_to_minio(import_file_path,file.filename)
        end_time=time.time()
        add_done_task(task_id,"upload_file")
        add_node_duration(task_id,"upload_file",end_time-start_time)


        #返回图片信息
        return task_id,import_file_path,file_dir

    def save_file_to_local(self, file_dir:str, file: UploadFile):
        """
        保存文件到临时目录
        Args:
            file_dir:文件上传目录
            file:文件上传对象

        Returns:import_file_path:str文件路径

        """
        #创建目录
        os.makedirs(file_dir,exist_ok=True)
        #文件路径
        import_file_path=os.path.join(file_dir,file.filename)

        #写入保存文件
        try:
            with open(file_dir, "wb") as f:
                shutil.copyfileobj(file.file, f)#不同的操作系统和python版本都可以分批次写入
        except IOError as e:
            logger.error(f"保存文件{file.filename}到临时目录失败，错误信息为:{e}")
            raise FileProcessingError(message=f"保存文件{file.filename}到临时目录失败",cause=e)

        #返回文件路径
        return import_file_path

    def save_file_to_minio(self, import_file_path:str, filename:str):
        """

        Args:
            import_file_path:上传文件地址
            filename:上传文件名字

        Returns:

        """
        #获得minio客户端
        try:
            minio_client=StorageClients.get_minio()
        except Exception as e:
            logger.error(f"获取minio客户端失败:{e}")
            return

        #获得相关信息
        bucket_name=os.getenv("MINIO_BUCKET_NAME")
        object_name=f"origin_file/{datetime.now().strftime('%Y%m%d')}/{filename}"
        #上传文件
        try:
            minio_client.fput_object(bucket_name,object_name,import_file_path)
        except Exception as e:
            logger.error(f"上传文件{filename}到minio失败:{e}")
