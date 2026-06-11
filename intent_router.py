"""
元数据提取器：LLM抽取查询中的元数据实现检索范围约束
意图路由器：数值精确查询 + 语义生成双通路架构
"""
import re
import json
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import config_data as config


class MetadataExtractor:
    """使用LLM从用户查询中提取元数据过滤条件"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个超市商品客服系统的元数据提取助手。
从用户查询中提取以下元数据字段（如果存在）：
- category: 商品类别（如：食品、日用品、服装、电器等）
- brand: 品牌名称
- price_range: 价格范围（如：0-50, 50-100, 100+）
- content_type: 内容类型（table 或 text）

只提取查询中明确提到的信息，不要猜测。
以JSON格式输出，只包含提取到的字段。

示例：
查询："推荐50元以下的食品"
输出：{{"category": "食品", "price_range": "0-50"}}

查询："这款Nike运动鞋的价格是多少"
输出：{{"brand": "Nike", "content_type": "table"}}
"""),
            ("user", "查询：{query}")
        ])
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def extract(self, query: str) -> dict:
        """提取查询元数据"""
        try:
            result = self.chain.invoke({"query": query})
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}


class IntentRouter:
    """意图路由：精准查询 + 语义生成双通路"""

    # 数值查询关键词
    NUMERIC_KEYWORDS = [
        "价格", "多少钱", "售价", "原价", "折扣", "打折", "优惠",
        "便宜", "贵", "性价比", "几折", "满减", "促销",
        "库存", "剩余", "销量", "数量",
        "重量", "规格", "容量", "尺寸", "大小"
    ]

    # 精确数值模式
    NUMERIC_PATTERNS = [
        r'\d+元', r'\d+块', r'\d+\.\d+元',
        r'低于\d+', r'高于\d+', r'超过\d+',
        r'\d+折', r'\d+%',
        r'价格.*\d+', r'\d+.*价格'
    ]

    def __init__(self):
        pass

    def is_numeric_query(self, query: str) -> bool:
        """判断是否为数值查询意图"""
        # 关键词匹配
        has_keyword = any(kw in query for kw in self.NUMERIC_KEYWORDS)

        # 正则模式匹配
        has_pattern = any(re.search(p, query) for p in self.NUMERIC_PATTERNS)

        return has_keyword or has_pattern

    def extract_numeric_constraints(self, query: str) -> dict:
        """提取数值约束条件"""
        constraints = {}

        # 提取价格范围
        price_match = re.search(r'(\d+(?:\.\d+)?)\s*元', query)
        if price_match:
            constraints["price"] = float(price_match.group(1))

        # 提取折扣
        discount_match = re.search(r'(\d+)\s*折', query)
        if discount_match:
            constraints["discount"] = int(discount_match.group(1)) / 10

        # 提取比较操作
        if any(kw in query for kw in ["低于", "小于", "不超过", "以下"]):
            constraints["operator"] = "lte"
        elif any(kw in query for kw in ["高于", "大于", "超过", "以上"]):
            constraints["operator"] = "gte"

        return constraints

    def route(self, query: str) -> tuple[str, dict]:
        """
        路由决策
        返回: (route_type, route_params)
        route_type: "numeric" 或 "semantic"
        """
        if self.is_numeric_query(query):
            constraints = self.extract_numeric_constraints(query)
            return "numeric", constraints
        return "semantic", {}


class NumericQueryHandler:
    """数值精确查询处理器"""

    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

        self.numeric_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是超市商品客服助手。请根据提供的参考资料回答问题。
对于价格、折扣等数值信息，请严格使用参考资料中的数据，不要推测。
参考资料：{context}
"""),
            ("user", "问题：{query}")
        ])
        self.chain = self.numeric_prompt | self.llm

    def search_exact(self, query: str, constraints: dict) -> list:
        """精确搜索包含数值信息的文档"""
        all_docs = self.vector_store.get()
        if not all_docs['ids']:
            return []

        # 过滤包含表格/数值特征的文档
        numeric_docs = []
        for i, metadata in enumerate(all_docs.get('metadatas', [])):
            if metadata.get('has_table') or metadata.get('content_type') == 'table':
                numeric_docs.append({
                    'id': all_docs['ids'][i],
                    'content': all_docs['documents'][i],
                    'metadata': metadata
                })

        # 如果没有专门的表格文档，返回所有文档
        if not numeric_docs:
            numeric_docs = [{
                'id': all_docs['ids'][i],
                'content': all_docs['documents'][i],
                'metadata': all_docs.get('metadatas', [{}]*len(all_docs['documents']))[i]
            } for i in range(len(all_docs.get('ids', [])))]

        return numeric_docs[:config.final_top_k]

    def handle(self, query: str, constraints: dict) -> str:
        """处理数值查询"""
        docs = self.search_exact(query, constraints)

        if not docs:
            return "抱歉，没有找到相关的价格信息。"

        context = "\n\n".join([doc['content'] for doc in docs])

        response = self.chain.invoke({
            "query": query,
            "context": context
        })

        return response.content if hasattr(response, 'content') else str(response)
