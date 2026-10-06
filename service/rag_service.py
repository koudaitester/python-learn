from langchain.vectorstores.base import VectorStore

# 注意：vector_db实例要在这里初始化，或者从外部传入，不要留在main.py
vector_db: VectorStore

def retrieve_knowledge(query: str, top_k=2) -> str:
    """检索知识库，返回拼接好的上下文文本"""
    docs = vector_db.similarity_search(query, k=top_k)
    context = "\n\n".join([f"【场景片段】{doc.page_content}" for doc in docs])
    return context

def build_llava_prompt(user_input: str, knowledge_context: str, image_desc: str|None = None) -> str:
    knowledge_context = retrieve_knowledge(user_input)
    prompt = f"""
        下面是参考知识库规则：
        {knowledge_context}

        根据上面的规则，分析图片信息：{image_desc}
        输出简短结论，判断环境是否存在风险，并给出对应的整改建议。
        要求：简洁直接，不要多余描述。
        """
    return prompt
