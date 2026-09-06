import threading
from typing import Optional
from langchain_openai import ChatOpenAI

from openai import OpenAI
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv()


class AIClients(BaseClientManager):
    """AI 模型类客户端：OpenAI(VLM)"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm_text_client: Optional[ChatOpenAI] = None
    _openai_llm_text_lock = threading.Lock()

    _openai_llm_json_client: Optional[ChatOpenAI] = None
    _openai_llm_json_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    _bge_rerank_client: Optional[FlagReranker] = None
    _bge_rerank_lock = threading.Lock()
    # ── OpenAI / VLM ──

    @classmethod
    def get_openai(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_openai)

    @classmethod
    def _create_openai(cls) -> OpenAI:
        try:
            api_key = cls._require_env("VLM_API_KEY")
            base_url = cls._require_env("VLM_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── ChatOpenAI / LLM text──

    @classmethod
    def get_llm_openai(cls,resonse_format:bool=True) -> ChatOpenAI:
        if resonse_format:
            # lambda传递的是一个匿名函数对象（就像一个还没执行的“操作说明书”）。
            return cls._get_or_create("_openai_llm_json_client", cls._openai_llm_json_lock,
                                      lambda: cls._create_llm_json_openai(resonse_format))
        else:
            return cls._get_or_create("_openai_llm_text_client", cls._openai_llm_text_lock_lock, cls._create_llm_text_openai)

    @classmethod
    def _create_llm_text_openai(cls) -> ChatOpenAI:
        try:
            api_key = cls._require_env("LLM_API_KEY")
            base_url = cls._require_env("LLM_API_BASE")
            model_name=cls._require_env("LLM_DEFAULT_MODEL")
            model_kwargs= {}

            llm_client =ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                model_kwargs=model_kwargs,
                extra_body={"enable_thinking": False}
            )

            logger.info(f"ChatOpenAI / LLM 客户端初始化成功 (base_url={base_url})")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI / LLM  客户端创建失败: {e}")
            raise ConnectionError(f"ChatOpenAI / LLM  连接失败: {e}") from e

    # ── ChatOpenAI / LLM json ──

    @classmethod
    def _create_llm_json_openai(cls, resonse_format: bool = True) -> ChatOpenAI:
        try:
            api_key = cls._require_env("LLM_API_KEY")
            base_url = cls._require_env("LLM_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")
            model_kwargs = {}
            # 输出解析器，控制llm输出为json模型
            if resonse_format:
                model_kwargs["response_format"] = {"type": "json_object"}
            llm_client = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                model_kwargs=model_kwargs,
                extra_body={"enable_thinking": False}
            )

            logger.info(f"ChatOpenAI / LLM 客户端初始化成功 (base_url={base_url})")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI / LLM  客户端创建失败: {e}")
            raise ConnectionError(f"ChatOpenAI / LLM  连接失败: {e}") from e
 # ── BGE-M3 ──

    @classmethod
    def get_bge_m3(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3", cls._bge_m3_lock, cls._create_bge_m3)

    @classmethod
    def _create_bge_m3(cls) -> BGEM3EmbeddingFunction:
        try:
            model_name = cls._require_env("BGE_M3_PATH")
            device = cls._require_env("BGE_DEVICE")
            fp16=cls._require_env("BGE_FP16")

            bge_m3_model=BGEM3EmbeddingFunction(
                model_name=model_name ,
                device=device,
                use_fp16=fp16,

            )
            return bge_m3_model
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE-M3客户端创建失败: {e}")
            raise ConnectionError(f"BGE-M3连接失败: {e}") from e

 # ── BGE-M3 reranker重排序模型 ──

    @classmethod
    def get_bge_reranker(cls) -> FlagReranker:
        return cls._get_or_create("_bge_reranker", cls._bge_rerank_lock, cls._create_bge_reranker)

    @classmethod
    def _create_bge_reranker(cls) -> FlagReranker:
        try:
            model_name = cls._require_env("BGE_RERANKER_LARGE")
            device = cls._require_env("BGE_RERANKER_DEVICE")
            fp16=cls._require_env("BGE_FP16")

            reranker=FlagReranker(
                model_name_or_path=model_name ,
                device=device,
                use_fp16=fp16,

            )
            return reranker
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE-M3重排序模型客户端创建失败: {e}")
            raise ConnectionError(f"BGE-M3重排序模型连接失败: {e}") from e
if __name__ == '__main__':
    llm_client=AIClients.get_llm_openai()
    print(llm_client)