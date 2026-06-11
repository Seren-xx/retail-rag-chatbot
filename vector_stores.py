"""
向量存储服务 + 知识库管理 + 混合检索
合并了向量存储、知识库、去重、文本分割等功能
"""
import os
import json
import chromadb
from datetime import datetime
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
import config_data as config
from text_splitter import AdvancedTextSplitter
from deduplication import DeduplicationService
from hybrid_retriever import HybridRetriever


class VectorStoreService:
    """向量存储服务"""

    def __init__(self, embedding):
        self.embedding = embedding
        os.makedirs(config.persist_directory, exist_ok=True)

        # ChromaDB 1.x 使用 PersistentClient
        chroma_client = chromadb.PersistentClient(path=config.persist_directory)

        self.vector_store = Chroma(
            client=chroma_client,
            collection_name=config.collection_name,
            embedding_function=self.embedding,
        )
        self.splitter = AdvancedTextSplitter()
        self.dedup = DeduplicationService(embedding_func=embedding)
        self.hybrid_retriever = None
        self._build_hybrid_retriever()

    def _build_hybrid_retriever(self):
        """构建混合检索器"""
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.vector_top_k}
        )
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            embedding_func=self.embedding
        )
        # 构建BM25索引
        try:
            all_docs = self.vector_store.get()
            if all_docs['ids']:
                documents = [
                    Document(
                        page_content=all_docs['documents'][i],
                        metadata=all_docs.get('metadatas', [{}]*len(all_docs['documents']))[i]
                    )
                    for i in range(len(all_docs['ids']))
                ]
                self.hybrid_retriever.build_bm25_index(documents)
        except Exception:
            pass

    def add_documents(self, texts_with_metadata: list[dict]) -> str:
        """
        添加文档到向量库
        texts_with_metadata: [{"text": "...", "metadata": {...}}, ...]
        """
        added_count = 0
        skipped_count = 0

        for item in texts_with_metadata:
            text = item["text"]
            metadata = item["metadata"]

            # 添加时间戳
            metadata["create_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 获取向量嵌入
            try:
                embedding = self.embedding.embed_query(text)
            except Exception:
                embedding = None

            # 三级去重检查
            chunk_id = f"{metadata.get('source', 'unknown')}_{metadata.get('chunk_index', 0)}"
            if self.dedup.is_duplicate(text, embedding):
                skipped_count += 1
                continue

            # 添加到向量库
            self.vector_store.add_texts(
                texts=[text],
                metadatas=[metadata]
            )

            # 添加到去重索引
            self.dedup.add(text, chunk_id, embedding)
            added_count += 1

        # 更新BM25索引
        self._build_hybrid_retriever()

        # 清理过期数据
        expired = self.dedup.remove_expired_data(self.vector_store)

        msg = f"[成功] 新增 {added_count} 条"
        if skipped_count > 0:
            msg += f", 跳过 {skipped_count} 条重复"
        if expired > 0:
            msg += f", 清理 {expired} 条过期数据"
        return msg

    def get_retriever(self):
        """返回混合检索器"""
        return self.hybrid_retriever

    def get_all_documents(self) -> dict:
        """获取所有文档"""
        return self.vector_store.get()

    def delete_documents(self, ids: list[str]):
        """删除文档"""
        self.vector_store.delete(ids=ids)

    def get_document_count(self) -> int:
        """获取文档数量"""
        try:
            all_docs = self.vector_store.get()
            return len(all_docs.get('ids', []))
        except Exception:
            return 0
