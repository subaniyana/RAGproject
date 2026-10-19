 1. 项目定位

  基于 FastAPI + LangGraph 的产品知识库问答系统，包含两条核心流水线：

  - 导入流水线：PDF/Markdown 文档 → 解析 → 切分 → 向量化 → 存入 Milvus
  - 查询流水线：用户提问 → 商品名确认 → 三路并行检索 → 融合重排 → LLM 生成答案（支持 SSE 流式输出）

  技术栈：FastAPI / LangGraph（工作流编排）/ Milvus（向量库，BGE-M3 混合检索）/ MinIO（对象存储）/ MongoDB（会话历史）/
  MinerU（PDF 解析）/ OpenAI 兼容 LLM + VLM / FlagEmbedding（BGE 重排）