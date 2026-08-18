# rag_query.py
import sys

import requests
from langchain_core.documents import Document

from search_vector_db import get_vectorstore

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:1b"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Use the provided context to answer "
    "questions. If you cannot find the answer in the context, say so. "
    "Only use information from the provided context."
)


def search_vectorstore(vectorstore, query: str, k: int = 3) -> list[Document]:
    """
    벡터 스토어에서 질의와 유사한 문서를 검색한다.

    Args:
        vectorstore: 벡터 스토어 인스턴스
        query (str): 검색 질의문
        k (int): 반환할 검색 결과 개수

    Returns:
        list[Document]: 질의와 관련된 문서 목록
    """
    return vectorstore.similarity_search(query, k=k)


def query_ollama(
    prompt: str,
    context: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    프롬프트와 문맥을 담아 Ollama 서버에 질의한다.

    Args:
        prompt (str): 사용자의 질문
        context (str): 벡터 스토어에서 검색한 문맥
        model (str): 사용할 모델 이름
        temperature (float): 생성 온도. 낮을수록 결정론적이다

    Returns:
        str: 생성된 답변
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Ollama 질의 실패: {e}"


def rag_query(
    vectorstore,
    query: str,
    num_contexts: int = 3,
) -> tuple[str, list[Document]]:
    """
    벡터 스토어와 Ollama를 이용해 RAG 질의를 수행한다.

    Args:
        vectorstore: 벡터 스토어 인스턴스
        query (str): 사용자의 질문
        num_contexts (int): 검색해 올 문맥 구절의 개수

    Returns:
        tuple[str, list[Document]]: (생성된 답변, 참조한 문서 목록)
    """
    relevant_docs = search_vectorstore(vectorstore, query, k=num_contexts)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    response = query_ollama(query, context)
    return response, relevant_docs

def query_ollama_plain(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """문맥 없이 질문만 보낸다. RAG 비교용."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Ollama 질의 실패: {e}"


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "What is LLMOps?"

    print(f"[질문] {query}")

    print("\n" + "=" * 50)
    print("RAG 없이 (모델이 아는 것만)")
    print("=" * 50)
    print(query_ollama_plain(query))

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