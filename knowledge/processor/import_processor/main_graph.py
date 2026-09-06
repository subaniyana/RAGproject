import logging
from langgraph.graph import END

from langgraph.graph.state import StateGraph, CompiledStateGraph

from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PDFToMDNode
from knowledge.processor.import_processor.nodes.md_img_node import MDToImgNode
from knowledge.processor.import_processor.nodes.import_milvus_node import MilvusImportNode
from knowledge.processor.import_processor.nodes.item_name_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode


from knowledge.processor.import_processor.state import ImportGraphState

"""
编排节点

定义节点
定义条件边
定义顺序边
运行整个pineline图片各个节点
"""

def import_router(state:ImportGraphState)->CompiledStateGraph:
    """
    根据其state中的属性is_pdf_read_enabled is_md_read_enable来决定如何到达下一个节点
    Returns:

    """
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node_zidingyi"
    elif state.get("is_md_read_enabled"):
        return "md_to_img_node_zidingyi"
    return END
def import_graph():
    """
    定义运行时的图状态workflow
    定义节点（入口节点，普通业务节点）
    定义边（条件便，普通业务边）
    返回运行时的状态
    Returns:

    """
    #构建状态管理流程图
    workflow=StateGraph(ImportGraphState)
    #定义入口节点
    workflow.set_entry_point("entry_node")
    #定义其它节点
    node_name_instance={
        "entry_node":EntryNode(),
        "pdf_to_md_node":PDFToMDNode(),
        "md_to_img_node":MDToImgNode(),
        "document_split_node":DocumentSplitNode(),
        "item_name_node":ItemNameRecognitionNode(),
        "embedding_chunks_node":EmbeddingChunksNode(),
        "import_milvus_node":MilvusImportNode(),
    }
    for node_name,node_instance in node_name_instance.items():
        workflow.add_node(node_name,node_instance)
    #定义边
    #定义条件边
    workflow.add_conditional_edges("entry_node",import_router,{
        "pdf_to_md_node_zidingyi":"pdf_to_md_node",#返回值：真实下一个节点名字
        "md_to_img_node_zidingyi":"md_to_img_node",
        END:END
    })
    #定义顺序边
    workflow.add_edge("pdf_to_md_node","md_to_img_node")
    workflow.add_edge("md_to_img_node","document_split_node")
    workflow.add_edge("document_split_node","item_name_node")
    workflow.add_edge("item_name_node","embedding_chunks_node")
    workflow.add_edge("embedding_chunks_node","import_milvus_node")
    workflow.add_edge("import_milvus_node",END)

    #编译
    compiled_workflow=workflow.compile()
    return compiled_workflow

import_app=import_graph()

def run_import_graph():
    #定义运行时的state
    state = ImportGraphState(
        {
         "import_file_path": r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用.pdf",
         "file_dir": r"E:\LLMlearn\daima\RAGproject\knowledge\processor\import_processor\temp_dir"
         }
    )
    for event in import_app.stream(state):
        final_state={}
        for key,value in event.items():
            print (f"当前执行的节点是{key}")
           # print(f"当前执行节点状态{value}")
            final_state=value
    return final_state

if __name__ == '__main__':
    from knowledge.processor.import_processor.base import setup_logging
    setup_logging(logging.INFO)  # 配置日志
    import json
    final_state=run_import_graph()
    print(json.dumps(final_state,ensure_ascii=False,indent=4))
    #得到状态图
    print(import_app.get_graph().print_ascii())