# 注释掉Chroma、OllamaEmbeddings相关全部代码
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
# embedding = OllamaEmbeddings(model="nomic-embed-text")
# vector_db = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embedding
# )

# 模拟RAG检索，固定返回一段测试知识库文本
def retrieve_knowledge(query: str, top_k=2) -> str:
    print(f"【模拟RAG检索】用户query：{query}")
    mock_context = """
【知识库规则】
1. 环境检测：地面有水渍，判定存在安全隐患。
2. 简短回答，不超过30个字。
"""
    return mock_context

def build_llava_prompt(user_input: str, knowledge_context: str, image_desc: str|None = None) -> str:
    prompt = f"""参考知识库规则:
{knowledge_context}
用户问题: {user_input}
请结合规则给出简短判断。
"""
    return prompt
