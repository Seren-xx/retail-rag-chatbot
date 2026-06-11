"""
超市商品智能客服系统（RAG增强）
基于Streamlit的统一Web应用：知识管理 + 智能问答
"""
import streamlit as st
import time
from rag import RagService
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config


# ============ 页面配置 ============
st.set_page_config(
    page_title="超市商品智能客服",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 超市商品智能客服系统")
st.caption("基于RAG增强 | 混合检索 | 智能路由")
st.divider()


# ============ 侧边栏 ============
with st.sidebar:
    st.header("系统管理")

    # 页面切换
    page = st.radio("选择功能", ["智能问答", "知识库管理", "系统状态"])

    st.divider()

    # 知识库统计
    if "vector_service" not in st.session_state:
        st.session_state["vector_service"] = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

    doc_count = st.session_state["vector_service"].get_document_count()
    st.metric("知识库文档数", doc_count)


# ============ 智能问答页面 ============
if page == "智能问答":
    st.subheader("💬 智能客服对话")

    # 初始化RAG服务
    if "rag" not in st.session_state:
        with st.spinner("初始化RAG服务..."):
            st.session_state["rag"] = RagService()

    # 对话历史
    if "message" not in st.session_state:
        st.session_state["message"] = [
            {"role": "assistant", "content": "你好！我是超市商品智能客服，可以帮您解答商品咨询、推荐及售后问题。请问有什么可以帮助您的？"}
        ]

    # 显示历史消息
    for message in st.session_state["message"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 用户输入
    prompt = st.chat_input("请输入您的问题...")

    if prompt:
        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state["message"].append({"role": "user", "content": prompt})

        # AI回复
        with st.chat_message("assistant"):
            with st.spinner("AI 思考中..."):
                try:
                    # 流式输出
                    ai_res_list = []
                    res_stream = st.session_state["rag"].chain.stream(
                        {"input": prompt},
                        config.session_config
                    )

                    def capture(generator, cache_list):
                        for chunk in generator:
                            cache_list.append(chunk)
                            yield chunk

                    st.write_stream(capture(res_stream, ai_res_list))
                    st.session_state["message"].append(
                        {"role": "assistant", "content": "".join(ai_res_list)}
                    )
                except Exception as e:
                    error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
                    st.error(error_msg)
                    st.session_state["message"].append(
                        {"role": "assistant", "content": error_msg}
                    )


# ============ 知识库管理页面 ============
elif page == "知识库管理":
    st.subheader("📚 知识库管理")

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传知识文档（支持txt格式）",
        type=['txt'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_size = uploaded_file.size / 1024

        st.info(f"文件名：{file_name} | 大小：{file_size:.2f} KB")

        # 读取文件内容
        text = uploaded_file.getvalue().decode("utf-8")

        if st.button("导入知识库", type="primary"):
            with st.spinner("正在处理文档..."):
                try:
                    # 使用高级分割器处理文本
                    from text_splitter import AdvancedTextSplitter
                    splitter = AdvancedTextSplitter()

                    # 分割文本
                    chunks = splitter.split_text(text)

                    # 添加元数据
                    texts_with_metadata = splitter.add_metadata(chunks, source=file_name)

                    # 添加到向量库
                    result = st.session_state["vector_service"].add_documents(texts_with_metadata)

                    st.success(result)
                except Exception as e:
                    st.error(f"导入失败：{str(e)}")

    st.divider()

    # 手动输入知识
    st.subheader("手动添加知识")
    with st.form("manual_input"):
        source_name = st.text_input("来源名称", placeholder="例如：尺码推荐指南")
        knowledge_text = st.text_area("知识内容", height=200, placeholder="请输入知识内容...")

        submitted = st.form_submit_button("添加", type="primary")

        if submitted and knowledge_text:
            with st.spinner("正在添加..."):
                try:
                    from text_splitter import AdvancedTextSplitter
                    splitter = AdvancedTextSplitter()

                    chunks = splitter.split_text(knowledge_text)
                    texts_with_metadata = splitter.add_metadata(chunks, source=source_name or "手动输入")

                    result = st.session_state["vector_service"].add_documents(texts_with_metadata)
                    st.success(result)
                except Exception as e:
                    st.error(f"添加失败：{str(e)}")


# ============ 系统状态页面 ============
elif page == "系统状态":
    st.subheader("📊 系统状态")

    # 知识库统计
    col1, col2 = st.columns(2)

    with col1:
        st.metric("知识库文档数", doc_count)

    with col2:
        st.metric("嵌入模型", config.embedding_model_name)

    st.divider()

    # 配置信息
    st.subheader("系统配置")
    config_data = {
        "向量检索TopK": config.vector_top_k,
        "BM25检索TopK": config.bm25_top_k,
        "RRF融合常数": config.rrf_k,
        "最终返回文档数": config.final_top_k,
        "CrossEncoder重排候选数": config.reranker_top_k,
        "文本块大小": config.chunk_size,
        "文本块重叠": config.chunk_overlap,
        "数据过期天数": config.data_expire_days,
        "SimHash汉明距离阈值": config.simhash_threshold,
        "向量语义相似度阈值": config.vector_simhash_threshold,
    }

    for key, value in config_data.items():
        st.text(f"{key}: {value}")

    st.divider()

    # 清空知识库
    if st.button("清空知识库", type="secondary"):
        if st.session_state.get("vector_service"):
            try:
                all_docs = st.session_state["vector_service"].get_all_documents()
                if all_docs['ids']:
                    st.session_state["vector_service"].delete_documents(all_docs['ids'])
                    st.success("知识库已清空")
                else:
                    st.info("知识库为空")
            except Exception as e:
                st.error(f"清空失败：{str(e)}")
