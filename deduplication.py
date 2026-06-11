"""
三级去重系统：MD5 + SimHash + 向量语义去重
结合时间戳元数据仅插入差异化内容，自动淘汰过期数据
"""
import os
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import List
import config_data as config


class MD5Deduplicator:
    """MD5精确去重"""

    def __init__(self):
        self.md5_path = config.md5_path
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.md5_path):
            open(self.md5_path, 'w', encoding='utf-8').close()

    def get_md5(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        md5_hex = self.get_md5(text)
        if not os.path.exists(self.md5_path):
            return False
        with open(self.md5_path, 'r', encoding='utf-8') as f:
            return md5_hex in f.read()

    def add(self, text: str):
        md5_hex = self.get_md5(text)
        with open(self.md5_path, 'a', encoding='utf-8') as f:
            f.write(md5_hex + '\n')


class SimHashDeduplicator:
    """SimHash近似去重"""

    def __init__(self):
        self.index_path = config.simhash_path
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _tokenize(self, text: str) -> list[str]:
        """简单中文分词"""
        return list(text)

    def _simhash(self, text: str) -> int:
        """计算文本的SimHash值"""
        tokens = self._tokenize(text)
        hash_vector = [0] * 64

        for token in tokens:
            token_hash = hashlib.md5(token.encode('utf-8')).hexdigest()
            binary_hash = bin(int(token_hash, 16))[2:].zfill(64)

            for i, bit in enumerate(binary_hash):
                if bit == '1':
                    hash_vector[i] += 1
                else:
                    hash_vector[i] -= 1

        simhash = 0
        for i in range(64):
            if hash_vector[i] > 0:
                simhash |= (1 << i)

        return simhash

    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """计算汉明距离"""
        xor = hash1 ^ hash2
        return bin(xor).count('1')

    def is_duplicate(self, text: str) -> bool:
        new_hash = self._simhash(text)
        for _, stored_hash in self.index.items():
            if self._hamming_distance(new_hash, stored_hash) <= config.simhash_threshold:
                return True
        return False

    def add(self, text: str, chunk_id: str):
        simhash_value = self._simhash(text)
        self.index[chunk_id] = simhash_value
        self._save_index()


class VectorSemanticDeduplicator:
    """向量语义去重：基于余弦相似度"""

    def __init__(self):
        self.index_path = config.vector_simhash_path
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def is_duplicate(self, embedding: list) -> bool:
        for _, stored_embedding in self.index.items():
            similarity = self._cosine_similarity(embedding, stored_embedding)
            if similarity >= config.vector_simhash_threshold:
                return True
        return False

    def add(self, chunk_id: str, embedding: list):
        self.index[chunk_id] = embedding
        self._save_index()


class DeduplicationService:
    """三级去重服务"""

    def __init__(self, embedding_func=None):
        self.md5_dedup = MD5Deduplicator()
        self.simhash_dedup = SimHashDeduplicator()
        self.vector_dedup = VectorSemanticDeduplicator()
        self.embedding_func = embedding_func

    def is_duplicate(self, text: str, embedding: list = None) -> bool:
        """三级去重检查"""
        # 第一级：MD5精确匹配
        if self.md5_dedup.is_duplicate(text):
            return True

        # 第二级：SimHash近似匹配
        if self.simhash_dedup.is_duplicate(text):
            return True

        # 第三级：向量语义匹配
        if embedding and self.vector_dedup.is_duplicate(embedding):
            return True

        return False

    def add(self, text: str, chunk_id: str, embedding: list = None):
        """添加到去重索引"""
        self.md5_dedup.add(text)
        self.simhash_dedup.add(text, chunk_id)
        if embedding:
            self.vector_dedup.add(chunk_id, embedding)

    def remove_expired_data(self, vector_store) -> int:
        """自动淘汰过期数据"""
        expire_date = datetime.now() - timedelta(days=config.data_expire_days)
        expire_str = expire_date.strftime("%Y-%m-%d %H:%M:%S")

        # 获取所有文档
        all_docs = vector_store.get()
        if not all_docs['ids']:
            return 0

        expired_ids = []
        for i, metadata in enumerate(all_docs['metadatas']):
            create_time = metadata.get("create_time", "")
            if create_time and create_time < expire_str:
                expired_ids.append(all_docs['ids'][i])

        if expired_ids:
            vector_store.delete(ids=expired_ids)

        return len(expired_ids)
