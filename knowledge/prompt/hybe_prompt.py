SYSTEM_HYPE_PROMPT_TEMPLATE = """您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"""
USER_HYDE_PROMPT_TEMPLATE = """
请根据用户的问题，编写一段技术文档片段作为回答参考。
商品名称：{item_names}
用户问题：{rewritten_query}

要求：
1. 文档片段应当专业、准确，使用技术文档的正式风格
2. 内容应紧扣用户问题，提供具体操作步骤或技术说明
3. 适当使用专业术语，但避免过于晦涩
4. 篇幅控制在 200-300 字左右
"""