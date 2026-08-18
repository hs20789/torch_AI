from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

EMBEDDING_MODEL = "text-embedding-3-small"
CONNECTION_STRING = "postgresql+psycopg://langchain:pass@localhost:5432/langchain"
TABLE_NAME = "pdf_docs"


def get_vectorstore() -> PGVectorStore:
    """기존 테이블에 연결한다. 테이블을 만들거나 지우지 않는다."""
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embeddings,
    )

def main() -> None:
    vectorstore = get_vectorstore()

    query = "MLOps란 무엇인가?"
    results = vectorstore.similarity_search_with_score(query, k=5)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- {i} (거리: {score:.4f}) ---")
        print(doc.page_content[:300])
        print("page:", doc.metadata.get("page"))

if __name__ == "__main__":
    main()