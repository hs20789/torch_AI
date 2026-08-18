# add_pdf.py
import sys
from pathlib import Path

import psycopg

from create_vector_db import (
    TABLE_NAME,
    load_pdf,
    split_documents,
    create_embeddings,
    create_engine,
    create_vectorstore,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# psycopg는 SQLAlchemy 문법(+psycopg)을 모르므로 별도 문자열이 필요하다
PSYCOPG_DSN = "postgresql://langchain:pass@localhost:5432/langchain"


def count_rows(table_name: str = TABLE_NAME) -> int:
    """테이블의 행 개수를 센다."""
    with psycopg.connect(PSYCOPG_DSN) as conn:
        row = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()
        return row[0]


def add_pdf(pdf_path: Path) -> None:
    """기존 테이블에 PDF를 추가한다. 테이블은 만들거나 지우지 않는다."""
    before = count_rows()

    texts = split_documents(load_pdf(pdf_path))
    print(f"추가할 청크: {len(texts)}")

    vectorstore = create_vectorstore(create_engine(), create_embeddings())
    vectorstore.add_documents(texts)

    after = count_rows()
    print(f"이전: {before} → 이후: {after}  (+{after - before})")


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python add_pdf.py <파일명.pdf>")
        sys.exit(1)

    pdf_path = DATA_DIR / sys.argv[1]
    if not pdf_path.exists():
        print(f"파일이 없습니다: {pdf_path}")
        sys.exit(1)

    add_pdf(pdf_path)


if __name__ == "__main__":
    main()
