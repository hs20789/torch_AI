"""PDF 인제스트. 몇 번을 실행해도 결과가 같다.

교재판(add_pdf.py)은 실행할 때마다 새 UUID로 청크를 다시 넣어 문서가
그대로 중복 적재된다. 그래서 검색 단계에서 중복을 걸러내는 코드가 필요했다.
여기서는 파일 해시로 변경 여부를 판단하고, 바뀐 문서만 통째로 교체한다.

  - 해시가 같다  -> 임베딩 API를 아예 호출하지 않고 건너뛴다 (비용 0)
  - 해시가 다르다 -> 해당 문서의 청크를 지우고 새로 넣는다 (한 트랜잭션)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import psycopg
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config, db, llm

# 이보다 짧은 청크는 머리말/쪽번호 같은 잡음일 가능성이 높다
MIN_CHUNK_CHARS = 40


@dataclass
class Chunk:
    """적재 단위."""

    chunk_id: str
    chunk_index: int
    page: int
    content: str


@dataclass
class IngestResult:
    """인제스트 결과 요약."""

    doc_id: str
    status: str  # skipped | created | updated
    n_chunks: int


def file_hash(path: Path) -> str:
    """
    파일 내용의 SHA-256 해시를 구한다.

    Args:
        path (Path): 파일 경로

    Returns:
        str: 16진수 해시 문자열
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    PDF를 쪽 단위로 읽는다. 빈 쪽은 버린다.

    Args:
        pdf_path (Path): PDF 파일 경로

    Returns:
        list[tuple[int, str]]: (쪽 번호, 본문) 목록. 쪽 번호는 1부터 센다
    """
    reader = PdfReader(str(pdf_path))
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((index, text))

    return pages


def chunk_pages(doc_id: str, pages: list[tuple[int, str]]) -> list[Chunk]:
    """
    쪽 단위로 청크를 만든다.

    쪽 경계를 넘지 않게 자르므로 출처(쪽 번호)가 정확해진다. 대신 쪽을 걸쳐
    이어지는 문장은 끊긴다. 표나 절 구조까지 살리려면 레이아웃 인식 파서
    (Docling, Unstructured 등)로 바꿔야 한다.

    Args:
        doc_id (str): 문서 식별자
        pages (list[tuple[int, str]]): (쪽 번호, 본문) 목록

    Returns:
        list[Chunk]: 청크 목록
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )

    chunks: list[Chunk] = []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue

            index = len(chunks)
            # 내용이 같으면 id도 같다. 실수로 두 번 넣어도 PK가 막아준다
            raw = f"{doc_id}|{index}|{piece}".encode()
            chunks.append(
                Chunk(
                    chunk_id=hashlib.sha256(raw).hexdigest()[:32],
                    chunk_index=index,
                    page=page_no,
                    content=piece,
                )
            )

    return chunks


def _stored_hash(conn: psycopg.Connection, doc_id: str) -> str | None:
    """저장된 문서 해시를 조회한다. 없으면 None."""
    with conn.cursor() as cur:
        cur.execute("SELECT doc_hash FROM rag_documents WHERE doc_id = %s", (doc_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _replace_document(
    conn: psycopg.Connection,
    doc_id: str,
    source_path: Path,
    doc_hash: str,
    tenant_id: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    """문서의 청크를 통째로 교체한다. 전부 성공하거나 전부 취소된다."""
    rows = [
        (
            chunk.chunk_id,
            doc_id,
            tenant_id,
            chunk.chunk_index,
            chunk.page,
            chunk.content,
            json.dumps({"source": source_path.name, "page": chunk.page}),
            db.to_pgvector(vector),
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM rag_chunks WHERE doc_id = %s", (doc_id,))
        cur.execute(
            """
            INSERT INTO rag_documents
                (doc_id, tenant_id, source_path, doc_hash, n_chunks, ingested_at)
            VALUES (%s, %s, %s, %s, %s, now())
            ON CONFLICT (doc_id) DO UPDATE SET
                tenant_id   = EXCLUDED.tenant_id,
                source_path = EXCLUDED.source_path,
                doc_hash    = EXCLUDED.doc_hash,
                n_chunks    = EXCLUDED.n_chunks,
                ingested_at = now()
            """,
            (doc_id, tenant_id, str(source_path), doc_hash, len(chunks)),
        )
        cur.executemany(
            """
            INSERT INTO rag_chunks
                (chunk_id, doc_id, tenant_id, chunk_index, page,
                 content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            rows,
        )


def ingest_pdf(
    conn: psycopg.Connection,
    pdf_path: Path,
    tenant_id: str = "default",
    force: bool = False,
) -> IngestResult:
    """
    PDF 한 건을 적재한다. 이미 같은 내용이 들어 있으면 건너뛴다.

    Args:
        conn (psycopg.Connection): 연결 객체
        pdf_path (Path): PDF 파일 경로
        tenant_id (str): 테넌트 식별자
        force (bool): True면 해시가 같아도 다시 적재한다

    Returns:
        IngestResult: 처리 결과
    """
    doc_id = pdf_path.name
    doc_hash = file_hash(pdf_path)
    previous = _stored_hash(conn, doc_id)

    if previous == doc_hash and not force:
        return IngestResult(doc_id=doc_id, status="skipped", n_chunks=0)

    chunks = chunk_pages(doc_id, load_pages(pdf_path))
    if not chunks:
        raise ValueError(f"추출된 텍스트가 없다: {pdf_path}")

    # 임베딩은 여기서 딱 한 번. 변경된 문서에만 비용이 든다
    vectors = llm.embed_texts([chunk.content for chunk in chunks])

    _replace_document(
        conn, doc_id, pdf_path, doc_hash, tenant_id, chunks, vectors
    )

    return IngestResult(
        doc_id=doc_id,
        status="updated" if previous else "created",
        n_chunks=len(chunks),
    )


def delete_document(conn: psycopg.Connection, doc_id: str) -> int:
    """
    문서와 그 청크를 지운다. 원본에서 삭제된 문서를 반영할 때 쓴다.

    Args:
        conn (psycopg.Connection): 연결 객체
        doc_id (str): 문서 식별자

    Returns:
        int: 삭제된 문서 행 수 (0 또는 1)
    """
    with conn.cursor() as cur:
        # rag_chunks는 ON DELETE CASCADE로 함께 지워진다
        cur.execute("DELETE FROM rag_documents WHERE doc_id = %s", (doc_id,))
        deleted = cur.rowcount
    conn.commit()
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF를 RAG 저장소에 적재한다")
    parser.add_argument(
        "files", nargs="*", help="data/ 안의 PDF 파일명. 생략하면 전부"
    )
    parser.add_argument("--tenant", default="default", help="테넌트 식별자")
    parser.add_argument(
        "--force", action="store_true", help="내용이 같아도 다시 적재한다"
    )
    parser.add_argument("--delete", metavar="DOC_ID", help="문서를 삭제한다")
    args = parser.parse_args()

    with db.connect() as conn:
        db.init_schema(conn)

        if args.delete:
            n = delete_document(conn, args.delete)
            print(f"삭제: {args.delete} ({n}건)")
            return

        if args.files:
            targets = [config.DATA_DIR / name for name in args.files]
        else:
            targets = sorted(config.DATA_DIR.glob("*.pdf"))

        for path in targets:
            if not path.exists():
                print(f"파일 없음: {path}")
                continue

            result = ingest_pdf(conn, path, tenant_id=args.tenant, force=args.force)
            label = {
                "skipped": "변경 없음, 건너뜀",
                "created": "신규 적재",
                "updated": "변경 감지, 교체",
            }[result.status]
            print(f"{result.doc_id}: {label} (청크 {result.n_chunks}개)")

        info = db.stats(conn, tenant_id=args.tenant)
        print(f"\n총 청크: {info['n_chunks']}개")


if __name__ == "__main__":
    main()
