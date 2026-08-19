"""설정값. 전부 환경 변수로 덮어쓸 수 있다."""

import os
from pathlib import Path

# --- 저장소 -------------------------------------------------------------
# psycopg는 SQLAlchemy 문법(+psycopg)을 쓰지 않는다
DSN = os.environ.get(
    "RAG_DSN", "postgresql://langchain:pass@localhost:5432/langchain"
)

# --- 모델 ---------------------------------------------------------------
EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
# 임베딩 모델을 바꾸면 차원도 바꾸고 "전체 재색인"을 해야 한다.
# 기존 벡터와 새 벡터는 서로 비교할 수 없다.
EMBEDDING_DIM = int(os.environ.get("RAG_EMBEDDING_DIM", "1536"))
EMBEDDING_BATCH = 100  # 한 요청에 보낼 청크 수

CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")
# RAG는 문맥에 충실해야 하므로 온도를 낮게 둔다 (교재판은 0.7)
CHAT_TEMPERATURE = float(os.environ.get("RAG_CHAT_TEMPERATURE", "0.2"))

# --- 청킹 ---------------------------------------------------------------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# --- 검색 ---------------------------------------------------------------
CANDIDATE_K = 30  # 벡터/키워드 각각의 1차 후보 수
TOP_K = 5  # 융합 후 LLM에 넘길 수
RRF_K = 60  # RRF 상수. 원 논문 권장값

# 코사인 거리 임계값. 0에 가까울수록 유사.
# 이 값보다 가까운 후보가 하나도 없으면 "문서에 근거가 없다"로 처리한다.
# 데이터마다 분포가 다르므로 evaluate.py로 반드시 보정할 것.
MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.62"))

# 문맥 길이 상한(문자). 토큰이 아니라 문자로 근사한다.
# 영어는 1토큰 ~4자, 한국어는 ~1.5자이므로 보수적으로 잡았다.
CONTEXT_CHAR_BUDGET = 6000

# --- 경로 ---------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
