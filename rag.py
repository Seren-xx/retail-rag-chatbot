"""
RAG服务：整合检索增强生成、意图路由、元数据提取
"""
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history
from intent_router import IntentRouter, NumericQueryHandler, MetadataExtractor


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagService:
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        # 意图路由
        self.intent_router = IntentRouter()
        self.numeric_handler = NumericQueryHandler(
            vector_store=self.vector_service.vector_store,
            llm=self.chat_model
        )

        # 元数据提取
        self.metadata_extractor = MetadataExtractor(llm=self.chat_model)

        # 提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是超市商品智能客服助手。以我提供的已知参考资料为主，"
                       "简洁和专业地回答用户问题。"
                       "如果参考资料中没有相关信息，请明确告知用户。"
                       "参考资料:{context}。"),
            ("system", "以下是用户的历史对话记录："),
            MessagesPlaceholder("history"),
            ("user", "请回答用户提问：{input}")
        ])

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def invoke_retriever(query: str) -> list[Document]:
            """包装混合检索器为可调用函数"""
            if retriever:
                return retriever.invoke(query)
            return []

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnableLambda(format_for_retriever) | RunnableLambda(invoke_retriever) | format_document
                } | RunnableLambda(
            format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

    def query(self, prompt: str, session_config: dict) -> str:
        """
        统一查询入口：自动路由到数值查询或语义查询
        """
        # 意图路由
        route_type, route_params = self.intent_router.route(prompt)

        if route_type == "numeric":
            # 数值精确查询通路
            return self.numeric_handler.handle(prompt, route_params)
        else:
            # 语义生成通路
            # 提取元数据用于检索约束
            metadata_filter = self.metadata_extractor.extract(prompt)

            # 使用混合检索器
            retriever = self.vector_service.get_retriever()
            if retriever:
                docs = retriever.invoke(prompt, metadata_filter=metadata_filter if metadata_filter else None)
            else:
                docs = []

            if not docs:
                return "抱歉，没有找到相关的商品信息。请尝试其他问题或联系人工客服。"

            # 构建上下文
            context = self._format_documents(docs)

            # 调用LLM生成回答
            response = self.chat_model.invoke(
                self.prompt_template.format_messages(
                    input=prompt,
                    context=context,
                    history=[]
                )
            )
            return response.content if hasattr(response, 'content') else str(response)

    def _format_documents(self, docs: list[Document]) -> str:
        """格式化文档为上下文"""
        if not docs:
            return "无相关参考资料"

        formatted_str = ""
        for doc in docs:
            formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

        return formatted_str


if __name__ == '__main__':
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = RagService().chain.invoke({"input": "我体重90斤，尺码推荐"}, session_config)
    print(res)
