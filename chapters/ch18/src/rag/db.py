"""PostgreSQL + pgvector 연결과 스키마.

프레임워크가 만들어주는 테이블을 쓰지 않고 직접 정의한다. 실무에서는
이 스키마가 곧 제품의 제약(멀티테넌시, 권한, 재색인)을 결정하기 때문에
숨겨두면 안 되는 부분이다.
"""

from __future__ import annotations

import psycopg

from . import config

# vector(__DIM__)은 init_schema에서 실제 차원으로 치환한다.
# ('{}'::jsonb 때문에 str.format을 쓸 수 없어 placeholder로 처리)
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- 원본 문서 1건 = 1행. doc_hash로 재수집 여부를 판단한다.
CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id      TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    source_path TEXT NOT NULL,
    doc_hash    TEXT NOT NULL,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES rag_documents(doc_id) ON DELETE CASCADE,
    -- 멀티테넌트/권한 필터의 자리. 이게 없으면 남의 문서가 답변에 섞인다.
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    chunk_index INTEGER NOT NULL,
    page        INTEGER,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(__DIM__) NOT NULL,
    -- 키워드 검색용. 'simple'은 어간 추출을 하지 않는 대신 언어를 가리지 않는다.
    -- 한국어 형태소 분석이 필요하면 pg_bigm이나 외부 분석기를 붙여야 한다.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
);
"""

# 인덱스는 따로 건다. HNSW는 pgvector 0.5.0 이상이 필요해서, 구버전이면
# 이 문장만 실패하고 나머지는 살아야 한다.
INDEX_SQL = [
    # ANN 인덱스. 없으면 순차 스캔이라 청크가 늘수록 선형으로 느려진다.
    "CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx "
    "ON rag_chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS rag_chunks_tsv_idx "
    "ON rag_chunks USING gin (tsv)",
    "CREATE INDEX IF NOT EXISTS rag_chunks_doc_idx ON rag_chunks (doc_id)",
    "CREATE INDEX IF NOT EXISTS rag_chunks_tenant_idx ON rag_chunks (tenant_id)",
]


def connect(dsn: str = config.DSN) -> psycopg.Connection:
    """
    데이터베이스에 연결한다.

    Args:
        dsn (str): 연결 문자열

    Returns:
        psycopg.Connection: 연결 객체
    """
    try:
        return psycopg.connect(dsn)
    except psycopg.OperationalError as e:
        # 실습용 CLI라 여기서 끊는다. 서비스라면 예외를 그대로 올려서
        # 상위 에러 핸들러가 재시도나 헬스체크 실패로 처리해야 한다.
        raise SystemExit(
            f"DB 연결 실패: {e}\n"
            "Postgres가 떠 있는지 확인할 것:\n"
            "  docker run -d --name pgvector -p 5432:5432 \\\n"
            "    -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=pass \\\n"
            "    -e POSTGRES_DB=langchain pgvector/pgvector:pg16"
        ) from e


def init_schema(conn: psycopg.Connection, dim: int = config.EMBEDDING_DIM) -> None:
    """
    테이블과 인덱스를 만든다. 이미 있으면 아무 일도 하지 않는다.

    실무에서는 이 DDL을 애플리케이션이 아니라 마이그레이션 도구(Alembic 등)가
    관리한다. 여기서는 실습 편의를 위해 함수로 둔다.

    Args:
        conn (psycopg.Connection): 연결 객체
        dim (int): 임베딩 벡터 차원
    """
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.replace("__DIM__", str(dim)))
    conn.commit()

    for statement in INDEX_SQL:
        try:
            with conn.cursor() as cur:
                cur.execute(statement)
            conn.commit()
        except psycopg.errors.Error as e:
            # 인덱스가 없어도 동작은 한다. 느려질 뿐이다
            conn.rollback()
            print(f"[경고] 인덱스 생성 실패: {e}")


def to_pgvector(vector: list[float]) -> str:
    """
    파이썬 실수 리스트를 pgvector 리터럴로 바꾼다.

    '[0.1,0.2,...]' 형태의 문자열을 넘기고 SQL에서 ::vector로 캐스팅한다.
    별도 어댑터 없이 동작하고, DB에 실제로 나가는 형태가 그대로 보인다.

    Args:
        vector (list[float]): 임베딩 벡터

    Returns:
        str: pgvector 리터럴 문자열
    """
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def stats(conn: psycopg.Connection, tenant_id: str = "default") -> dict:
    """
    적재 현황을 조회한다.

    Args:
        conn (psycopg.Connection): 연결 객체
        tenant_id (str): 테넌트 식별자

    Returns:
        dict: 문서 수, 청크 수, 문서별 청크 수
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, n_chunks, ingested_at FROM rag_documents "
            "WHERE tenant_id = %s ORDER BY doc_id",
            (tenant_id,),
        )
        docs = cur.fetchall()
        cur.execute(
            "SELECT count(*) FROM rag_chunks WHERE tenant_id = %s", (tenant_id,)
        )
        n_chunks = cur.fetchone()[0]

    return {"documents": docs, "n_chunks": n_chunks}


def main() -> None:
    with connect() as conn:
        init_schema(conn)
        info = stats(conn)

    print(f"스키마 준비 완료. 청크 {info['n_chunks']}개")
    for doc_id, n, at in info["documents"]:
        print(f"  - {doc_id}: {n}개 청크 ({at:%Y-%m-%d %H:%M})")


if __name__ == "__main__":
    main()
