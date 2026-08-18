# rag_query_gpt.py
import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from search_vector_db import get_vectorstore

DEFAULT_MODEL = "gpt-4o-mini"

# 큰 모델은 문맥에만 가두지 않고 조금 자유롭게 두는 편이 낫다 (책 18.3.1)
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Use the following context to answer "
        "questions. Please provide as much detail as possible in a "
        "comprehensive answer.",
    ),
    ("system", "Context:\n{context}"),
    ("user", "{question}"),
])


def search_vectorstore(vectorstore, query: str, k: int = 3) -> list[Document]:
    """
    벡터 스토어에서 질의와 유사한 문서를 검색하고 중복을 제거한다.

    Args:
        vectorstore: 벡터 스토어 인스턴스
        query (str): 검색 질의문
        k (int): 반환할 검색 결과 개수

    Returns:
        list[Document]: 중복이 제거된 관련 문서 목록
    """
    # 중복이 섞일 것을 감안해 더 많이 요청한 뒤 걸러낸다
    results = vectorstore.similarity_search(query, k=k * 2)

    seen: set[int] = set()
    unique: list[Document] = []

    for doc in results:
        content_hash = hash(doc.page_content.strip())
        if content_hash in seen:
            continue

        seen.add(content_hash)
        unique.append(doc)
        if len(unique) == k:
            break

    return unique


def query_gpt(
    prompt: str,
    context: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    프롬프트와 문맥을 담아 GPT에 질의한다.

    Args:
        prompt (str): 사용자의 질문
        context (str): 벡터 스토어에서 검색한 문맥
        model (str): 사용할 모델 이름
        temperature (float): 생성 온도

    Returns:
        str: 생성된 답변
    """
    chat = ChatOpenAI(model=model, temperature=temperature)
    formatted = RAG_PROMPT.format(context=context, question=prompt)
    return chat.invoke(formatted).content


def query_gpt_plain(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    문맥 없이 질문만 보낸다. RAG 비교용.

    Args:
        prompt (str): 사용자의 질문
        model (str): 사용할 모델 이름
        temperature (float): 생성 온도

    Returns:
        str: 생성된 답변
    """
    chat = ChatOpenAI(model=model, temperature=temperature)
    return chat.invoke(prompt).content


def rag_query(
    vectorstore,
    query: str,
    num_contexts: int = 3,
) -> tuple[str, list[Document]]:
    """
    벡터 스토어와 GPT를 이용해 RAG 질의를 수행한다.

    Args:
        vectorstore: 벡터 스토어 인스턴스
        query (str): 사용자의 질문
        num_contexts (int): 검색해 올 문맥 구절의 개수

    Returns:
        tuple[str, list[Document]]: (생성된 답변, 참조한 문서 목록)
    """
    relevant_docs = search_vectorstore(vectorstore, query, k=num_contexts)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    response = query_gpt(query, context)
    return response, relevant_docs


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "What is LLMOps?"

    print(f"[질문] {query}")

    print("\n" + "=" * 50)
    print("RAG 없이 (모델이 아는 것만)")
    print("=" * 50)
    print(query_gpt_plain(query))

    vectorstore = get_vectorstore()
    answer, sources = rag_query(vectorstore, query, num_contexts=3)

    print("\n" + "=" * 50)
    print("RAG 사용 (문서 근거)")
    print("=" * 50)
    print(answer)

    print("\n[출처]")
    for i, doc in enumerate(sources, 1):
        print(f"{i}. page {doc.metadata.get('page')}")


if __name__ == "__main__":
    main()