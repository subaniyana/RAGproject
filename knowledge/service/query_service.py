import uuid
import logging
from typing import List, Dict, Any

from torch.distributed.elastic.multiprocessing.errors import record

from knowledge.processor.query_processor.main_graph import create_query_graph
from knowledge.utils.mongo_history_util import clear_history
from knowledge.utils.task_util import *
from knowledge.utils.mongo_history_util import get_recent_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryService:

    @staticmethod
    def generate_session_id():
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id():
        return str(uuid.uuid4().hex[:8])

    def run_query_graph(self, task_id: str,  session_id: str,query: str, is_stream: bool):
        """
        运行查询图
        param task_id:任务id
        param session_id:会话id
        param query:查询问题
        is_stream:是否流式
        :return:
        """
        #修改任务状态
        update_task_status(task_id,TASK_STATUS_PROCESSING)
        #构建查询初始化状态
        query_init_state={
            "session_id":session_id,
            "query":query,
            "is_stream":is_stream,
            "original_query":query
        }
        try:
            #运行查询图,graph图谱的调用stream与invoke与is_stream无关，is_stream只是影响llm调用
            query_graph = create_query_graph()
            query_graph.invoke(query_init_state)

            #更新任务状态为完成状态
            update_task_status(task_id,TASK_STATUS_COMPLETED)

        except Exception as e:
            logger.error(f"运行graph流程图出现异常{e}")
            #更新任务状态为失败状态
            update_task_status(task_id,TASK_STATUS_FAILED)


    def get_history(self, session_id:str, limit:int)->List[Dict[str,Any]]:
        records=get_recent_messages(session_id,limit)
        return[
            {
                "_id":str(record.get("_id","")),
                "session_id":record.get("session_id",""),
                "role":record.get("role",""),
                "text":record.get("text",""),
                "rewritten_query":record.get("rewritten_query",""),
                "item_names":record.get("item_names",""),
                "ts":record.get("ts",""),
            }
            for record in records
        ]


    def delete_history(self, session_id):
        return clear_history(session_id)
