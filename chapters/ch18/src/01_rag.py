# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
from pathlib import Path

# %%
import os

print(os.environ.get("OPENAI_API_KEY") is not None)

# %%
# pdf 파일을 로드한다.
pdf_path = Path(__file__).parent.parent / "data" / "example.pdf"
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

# %%
# 문서를 청크로 나눈다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.split_documents(documents)
# %%
# 오픈 AI 임베딩을 초기화
# 환경 변수 OPENAI_API_KEY를 설정해야한다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
VECTOR_SIZE = 1536  # text-embedding-3-small의 차원

# 벡터 저장소를 생성하고 유지한다.
CONNECTION_STRING = "postgresql+psycopg://langchain:pass@localhost:5432/langchain"

engine = PGEngine.from_connection_string(url=CONNECTION_STRING)


# %%
print(len(texts))

# %%
# 벡터 테이블 생성 (최초 1회)
TABLE_NAME = "pdf_docs"

engine.init_vectorstore_table(
    table_name=TABLE_NAME,
    vector_size=VECTOR_SIZE,
    overwrite_existing=True,  # 셀 재실행 시 테이블을 다시 만든다
)

# %%
# 벡터 저장소를 생성하고 문서를 넣는다.
vectorstore = PGVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
    embedding_service=embeddings,
)

vectorstore.add_documents(texts)
