
# ============ 文件存储 ============
md5_path = "./md5.text"
simhash_path = "./simhash_index.json"
vector_simhash_path = "./vector_simhash_index.json"

# ============ Chroma向量库 ============
collection_name = "rag"
persist_directory = "./chroma_db"

# ============ 文本分割配置 ============
# 段落优先+标点分句+块重叠策略
chunk_size = 500
chunk_overlap = 80
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""]
max_split_char_number = 500  # 超过此长度才分割

# ============ 模型配置 ============
embedding_model_name = "bge-m3"  # 改用BGE-M3
chat_model_name = "qwen3-max"
reranker_model_name = "bge-reranker-v2-m3"  # CrossEncoder重排模型

# ============ 检索配置 ============
# 混合检索
vector_top_k = 10       # 向量检索返回数量
bm25_top_k = 10         # BM25关键词检索返回数量
rrf_k = 60              # RRF融合常数
final_top_k = 5         # 最终返回文档数
reranker_top_k = 10     # CrossEncoder重排候选数

# 语义相似度阈值
similarity_threshold = 0.7

# ============ 去重配置 ============
md5_threshold = 1.0             # MD5完全匹配
simhash_threshold = 3           # SimHash汉明距离阈值
vector_simhash_threshold = 0.85 # 向量余弦相似度阈值

# ============ 数据过期配置 ============
data_expire_days = 90  # 数据过期天数

# ============ Session配置 ============
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}

