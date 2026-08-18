import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 기본 설정값
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536  # text-embedding-3-small의 차원
CONNECTION_STRING = "postgresql+psycopg://langchain:pass@localhost:5432/langchain"
TABLE_NAME = "pdf_docs"


def check_api_key() -> bool:
    """
    OPENAI_API_KEY 환경 변수가 설정되어 있는지 확인한다.

    Returns:
        bool: 환경 변수가 설정되어 있으면 True
    """
    return os.environ.get("OPENAI_API_KEY") is not None


def load_pdf(pdf_path: Path) -> list[Document]:
    """
    PDF 파일을 로드해 페이지 단위 문서로 반환한다.

    Args:
        pdf_path (Path): PDF 파일 경로

    Returns:
        list[Document]: 로드된 문서 목록
    """
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    문서를 청크로 나눈다.

    Args:
        documents (list[Document]): 나눌 문서 목록
        chunk_size (int): 청크 하나의 최대 길이
        chunk_overlap (int): 이웃한 청크끼리 겹치는 길이

    Returns:
        list[Document]: 청크로 나뉜 문서 목록
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def create_embeddings(model: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:
    """
    오픈 AI 임베딩을 초기화한다.
    환경 변수 OPENAI_API_KEY를 설정해야한다.

    Args:
        model (str): 사용할 임베딩 모델 이름

    Returns:
        OpenAIEmbeddings: 임베딩 인스턴스
    """
    return OpenAIEmbeddings(model=model)


def create_engine(connection_string: str = CONNECTION_STRING) -> PGEngine:
    """
    PostgreSQL 연결 엔진을 만든다.

    Args:
        connection_string (str): 데이터베이스 연결 문자열

    Returns:
        PGEngine: 연결 엔진 인스턴스
    """
    return PGEngine.from_connection_string(url=connection_string)


def init_vector_table(
    engine: PGEngine,
    table_name: str = TABLE_NAME,
    vector_size: int = VECTOR_SIZE,
    overwrite_existing: bool = True,
) -> None:
    """
    벡터 테이블을 생성한다. (최초 1회)

    Args:
        engine (PGEngine): 연결 엔진 인스턴스
        table_name (str): 생성할 테이블 이름
        vector_size (int): 임베딩 벡터의 차원
        overwrite_existing (bool): True면 재실행 시 테이블을 다시 만든다
    """
    engine.init_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        overwrite_existing=overwrite_existing,
    )


def create_vectorstore(
    engine: PGEngine,
    embeddings: OpenAIEmbeddings,
    table_name: str = TABLE_NAME,
) -> PGVectorStore:
    """
    벡터 저장소를 생성한다.

    Args:
        engine (PGEngine): 연결 엔진 인스턴스
        embeddings (OpenAIEmbeddings): 임베딩 인스턴스
        table_name (str): 사용할 테이블 이름

    Returns:
        PGVectorStore: 벡터 저장소 인스턴스
    """
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=table_name,
        embedding_service=embeddings,
    )


def build_vectorstore(
    pdf_path: Path,
    table_name: str = TABLE_NAME,
    connection_string: str = CONNECTION_STRING,
) -> PGVectorStore:
    """
    PDF 파일로부터 벡터 저장소를 만들고 문서를 넣는다.

    Args:
        pdf_path (Path): PDF 파일 경로
        table_name (str): 사용할 테이블 이름
        connection_string (str): 데이터베이스 연결 문자열

    Returns:
        PGVectorStore: 문서가 저장된 벡터 저장소 인스턴스
    """
    documents = load_pdf(pdf_path)
    texts = split_documents(documents)
    print(len(texts))

    embeddings = create_embeddings()
    engine = create_engine(connection_string)
    init_vector_table(engine, table_name=table_name)

    vectorstore = create_vectorstore(engine, embeddings, table_name=table_name)
    vectorstore.add_documents(texts)

    return vectorstore


def main() -> None:
    print(check_api_key())

    pdf_path = Path(__file__).parent.parent / "data" / "example.pdf"
    build_vectorstore(pdf_path)


if __name__ == "__main__":
    main()
