"""
高级文本分割器：段落优先+标点分句+块重叠策略
解决固定长度切分造成的语义断裂问题
"""
import re
import config_data as config


class AdvancedTextSplitter:
    """段落优先+标点分句+块重叠的文本分割器"""

    def __init__(self):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.separators = config.separators

    def split_by_paragraphs(self, text: str) -> list[str]:
        """按段落分割文本"""
        paragraphs = re.split(r'\n\n+', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def split_by_sentences(self, text: str) -> list[str]:
        """按标点符号分句"""
        # 使用配置的分割符，优先中文标点
        pattern = '|'.join(re.escape(s) for s in self.separators if len(s) > 0)
        if not pattern:
            return [text]

        sentences = re.split(f'({pattern})', text)
        result = []
        current = ""
        for part in sentences:
            if part in self.separators:
                if current.strip():
                    result.append(current.strip() + part)
                    current = ""
            else:
                current += part
        if current.strip():
            result.append(current.strip())
        return result

    def merge_chunks_with_overlap(self, sentences: list[str]) -> list[str]:
        """将句子合并成块，保持重叠"""
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 保留重叠部分
                if len(sentence) <= self.chunk_overlap:
                    current_chunk = sentence
                else:
                    # 取句子末尾作为重叠
                    overlap_start = max(0, len(sentence) - self.chunk_overlap)
                    current_chunk = sentence[overlap_start:] + sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def split_text(self, text: str) -> list[str]:
        """
        段落优先+标点分句+块重叠策略
        1. 先按段落分割
        2. 对过长段落按标点分句
        3. 合并句子成块并保留重叠
        """
        if len(text) <= config.max_split_char_number:
            return [text]

        # 1. 段落分割
        paragraphs = self.split_by_paragraphs(text)

        all_chunks = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                all_chunks.append(para)
            else:
                # 2. 过长段落按标点分句
                sentences = self.split_by_sentences(para)
                # 3. 合并成块并保留重叠
                chunks = self.merge_chunks_with_overlap(sentences)
                all_chunks.extend(chunks)

        return all_chunks

    def add_metadata(self, chunks: list[str], source: str) -> list[dict]:
        """为文本块添加多维元数据"""
        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_length": len(chunk),
            }
            # 检测是否包含表格特征
            if any(keyword in chunk for keyword in ["|", "价格", "元", "折扣", "￥", "$"]):
                metadata["has_table"] = True
                metadata["content_type"] = "table"
            else:
                metadata["has_table"] = False
                metadata["content_type"] = "text"

            # 检测内容类型
            if any(keyword in chunk for keyword in ["尺码", "尺寸", "大小", "推荐"]):
                metadata["category"] = "size_guide"
            elif any(keyword in chunk for keyword in ["颜色", "色彩", "色差"]):
                metadata["category"] = "color_guide"
            elif any(keyword in chunk for keyword in ["洗涤", "养护", "清洗", "保养"]):
                metadata["category"] = "care_guide"
            else:
                metadata["category"] = "general"

            metadata_list.append({"text": chunk, "metadata": metadata})

        return metadata_list
