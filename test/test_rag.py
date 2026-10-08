from services.rag_service import retrieve_knowledge, build_llava_prompt

def test_rag():
    # 测试检索知识库
    query = "测试风险检查"
    ctx = retrieve_knowledge(query)
    print("====检索出来的知识库上下文====")
    print(ctx)

    # 组装prompt
    prompt = build_llava_prompt(user_input=query, knowledge_context=ctx)
    print("\n====最终组装好的Prompt====")
    print(prompt)

if __name__ == "__main__":
    test_rag()