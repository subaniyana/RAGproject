from functools import cache, lru_cache

from knowledge.service.query_service import QueryService
from knowledge.service.upload_service import UploadService



@lru_cache#缓存注解，将实例对象缓存一份，根据lru算法淘汰一些缓存
def get_upload_file_service():
    return UploadService

@lru_cache#缓存注解，将实例对象缓存一份，根据lru算法淘汰一些缓存
def get_query_service():
    return QueryService