"""
混合检索器：向量检索 + BM25关键词检索 + RRF融合 + CrossEncoder重排
"""
import re
from typing import List
from langchain_core.documents import Document
import config_data as config


class BM25Retriever:
    """BM25关键词检索器"""

    def __init__(self):
        self.documents = []
        self.doc_freq = {}  # 词项 -> 包含该词的文档数
        self.doc_terms = []  # 每个文档的词项列表
        self.avg_doc_len = 0
        self.k1 = 1.5
        self.b = 0.75
        self.is_built = False

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：中文按字，英文按词"""
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        # 中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return english_words + chinese_chars

    def build_index(self, documents: List[Document]):
        """构建BM25索引"""
        self.documents = documents
        self.doc_freq = {}
        self.doc_terms = []

        for doc in documents:
            terms = self._tokenize(doc.page_content)
            self.doc_terms.append(terms)
            # 统计文档频率
            unique_terms = set(terms)
            for term in unique_terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.avg_doc_len = sum(len(terms) for terms in self.doc_terms) / max(len(documents), 1)
        self.is_built = True

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        """计算单个文档的BM25分数"""
        terms = self.doc_terms[doc_idx]
        doc_len = len(terms)
        score = 0.0
        n_docs = len(self.documents)

        for term in query_terms:
            if term not in self.doc_freq:
                continue

            tf = terms.count(term)
            df = self.doc_freq[term]

            idf = max(0, (n_docs - df + 0.5) / (df + 0.5))

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)

            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = None) -> List[Document]:
        """BM25检索"""
        if not self.is_built:
            return []

        top_k = top_k or config.bm25_top_k
        query_terms = self._tokenize(query)

        scores = []
        for i in range(len(self.documents)):
            score = self._score(query_terms, i)
            scores.append((i, score))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)

        return [self.documents[idx] for idx, _ in scores[:top_k]]


class HybridRetriever:
    """混合检索器：向量 + BM25 + RRF融合 + CrossEncoder重排"""

    def __init__(self, vector_retriever, embedding_func=None):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = BM25Retriever()
        self.embedding_func = embedding_func
        self.reranker = None
        self._all_documents = []

    def build_bm25_index(self, documents: List[Document]):
        """构建BM25索引"""
        self._all_documents = documents
        self.bm25_retriever.build_index(documents)

    def _rrf_merge(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = None) -> List[Document]:
        """RRF（Reciprocal Rank Fusion）融合"""
        k = k or config.rrf_k

        # 文档ID -> (doc, score)
        doc_scores = {}

        # 向量检索结果
        for rank, doc in enumerate(vector_docs):
            doc_id = id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = (doc, 0.0)
            doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + 1.0 / (k + rank + 1))

        # BM25检索结果
        for rank, doc in enumerate(bm25_docs):
            doc_id = id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = (doc, 0.0)
            doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + 1.0 / (k + rank + 1))

        # 按RRF分数排序
        ranked_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs]

    def _crossencoder_rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """CrossEncoder重排"""
        if not documents or self.reranker is None:
            return documents[:top_k or config.final_top_k]

        try:
            from sentence_transformers import CrossEncoder
            if self.reranker is None:
                self.reranker = CrossEncoder(config.reranker_model_name)

            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.reranker.predict(pairs)

            # 按分数排序
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return [doc for doc, _ in scored_docs[:top_k or config.final_top_k]]
        except Exception:
            # 如果CrossEncoder不可用，降级使用原始排序
            return documents[:top_k or config.final_top_k]

    def invoke(self, query: str, metadata_filter: dict = None) -> List[Document]:
        """混合检索入口"""
        # 1. 向量检索
        vector_docs = self.vector_retriever.invoke(query)

        # 2. BM25关键词检索
        bm25_docs = self.bm25_retriever.search(query)

        # 3. RRF融合
        merged_docs = self._rrf_merge(vector_docs, bm25_docs)

        # 4. 应用元数据过滤
        if metadata_filter:
            filtered_docs = []
            for doc in merged_docs:
                match = True
                for key, value in metadata_filter.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_docs.append(doc)
            merged_docs = filtered_docs

        # 5. CrossEncoder重排
        reranked_docs = self._crossencoder_rerank(query, merged_docs)

        return reranked_docs[:config.final_top_k]
