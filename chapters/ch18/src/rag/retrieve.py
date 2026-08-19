"""하이브리드 검색.

교재판은 similarity_search(query, k=3) 한 줄이었다. 벡터 검색만 쓰면
의미는 잘 잡지만 제품명, 약어, 에러 코드 같은 정확 매칭에 약하다.
("pgvector", "Canary", "ORA-01555" 같은 것들)

그래서 두 갈래로 뽑아 RRF로 합친다.

    벡터 검색 (의미)   ─┐
                        ├─ RRF 융합 ─ 거리 임계값 검사 ─ 문맥 조립
    키워드 검색 (형태) ─┘

리랭커(cross-encoder)를 붙이면 여기서 한 단계 더 올라가지만 별도 모델이
필요하므로 이 파일에는 넣지 않았다.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field

import psycopg

from . import config, db, llm

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")

# 조사를 대충 떼어낸다. 제대로 하려면 형태소 분석기(Kiwi, mecab)를 쓰거나
# pg_bigm으로 n-gram 색인을 걸어야 한다. 여기서는 접두 검색(:*)과 함께 써서
# "LLMOps는", "LLMOps를" 같은 변형을 한 번에 잡는 정도로만 쓴다.
_JOSA = (
    "에서는", "으로는", "에게서", "이라는", "라는", "에서", "으로", "에게",
    "까지", "부터", "한테", "보다", "처럼", "마다", "이나", "와의", "과의",
    "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "로", "만", "에",
)


# 'simple' 사전에는 불용어가 없어서 "what", "is", "무엇" 같은 토큰이 거의 모든
# 청크에 걸린다. ts_rank_cd가 어느 정도 걸러주지만 미리 빼는 편이 깔끔하다.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "what", "how", "why", "when", "where", "who", "which",
    "do", "does", "did", "it", "this", "that", "with", "as", "by", "at", "from",
    "무엇", "무엇인", "어떻게", "어떤", "어떻", "언제", "어디", "누가", "누구",
    "그리고", "그래서", "하는", "한다", "인가", "있나", "이란", "대해", "관해",
}


@dataclass
class Hit:
    """검색된 청크 하나."""

    chunk_id: str
    doc_id: str
    page: int
    content: str
    distance: float | None = None  # 코사인 거리. 작을수록 유사
    keyword_rank: int | None = None  # 키워드 검색 순위
    score: float = 0.0  # RRF 융합 점수
    found_by: set[str] = field(default_factory=set)


def _stem(token: str) -> str:
    """한글 토큰의 조사를 하나 떼어낸다."""
    if not re.search(r"[가-힣]", token):
        return token

    for josa in _JOSA:
        if token.endswith(josa) and len(token) - len(josa) >= 2:
            return token[: -len(josa)]
    return token


def to_tsquery(query: str) -> str:
    """
    질의문을 tsquery 문자열로 바꾼다.

    plainto_tsquery는 모든 단어를 AND로 묶어서 질문형 문장에는 거의 걸리지
    않는다. 여기서는 OR로 풀고 접두 검색을 붙인다.
    "LLMOps는 무엇인가" -> "llmops:* | 무엇:*"

    Args:
        query (str): 사용자 질의문

    Returns:
        str: to_tsquery에 넘길 문자열. 유효한 토큰이 없으면 빈 문자열
    """
    seen: list[str] = []
    for token in _TOKEN_RE.findall(query.lower()):
        stem = _stem(token)
        if token in _STOPWORDS or stem in _STOPWORDS:
            continue
        if len(stem) >= 2 and stem not in seen:
            seen.append(stem)

    return " | ".join(f"{stem}:*" for stem in seen)


def vector_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    tenant_id: str = "default",
    limit: int = config.CANDIDATE_K,
) -> list[tuple]:
    """
    코사인 거리로 유사한 청크를 찾는다.

    Args:
        conn (psycopg.Connection): 연결 객체
        query_vector (list[float]): 질의 임베딩
        tenant_id (str): 테넌트 식별자
        limit (int): 후보 개수

    Returns:
        list[tuple]: (chunk_id, doc_id, page, content, distance) 목록
    """
    literal = db.to_pgvector(query_vector)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, doc_id, page, content,
                   embedding <=> %s::vector AS distance
            FROM rag_chunks
            WHERE tenant_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (literal, tenant_id, literal, limit),
        )
        return cur.fetchall()


def keyword_search(
    conn: psycopg.Connection,
    query: str,
    tenant_id: str = "default",
    limit: int = config.CANDIDATE_K,
) -> list[tuple]:
    """
    전문 검색으로 청크를 찾는다.

    Args:
        conn (psycopg.Connection): 연결 객체
        query (str): 사용자 질의문
        tenant_id (str): 테넌트 식별자
        limit (int): 후보 개수

    Returns:
        list[tuple]: (chunk_id, doc_id, page, content, rank) 목록
    """
    tsquery = to_tsquery(query)
    if not tsquery:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.page, c.content,
                   ts_rank_cd(c.tsv, q) AS rank
            FROM rag_chunks c, to_tsquery('simple', %s) AS q
            WHERE c.tenant_id = %s AND c.tsv @@ q
            ORDER BY rank DESC
            LIMIT %s
            """,
            (tsquery, tenant_id, limit),
        )
        return cur.fetchall()


def rrf_fuse(
    ranked_lists: dict[str, list[str]], k: int = config.RRF_K
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion으로 여러 순위 목록을 합친다.

    점수 스케일이 다른 검색기(거리 vs ts_rank)를 정규화 없이 합칠 수 있다.
    순위만 보기 때문이다. score = sum(1 / (k + rank))

    Args:
        ranked_lists (dict[str, list[str]]): 검색기 이름 -> 순위대로 정렬된 id 목록
        k (int): 완충 상수. 클수록 상위권 가중치가 완만해진다

    Returns:
        dict[str, float]: chunk_id -> 융합 점수
    """
    scores: dict[str, float] = defaultdict(float)

    for ids in ranked_lists.values():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] += 1.0 / (k + rank)

    return scores


def retrieve(
    conn: psycopg.Connection,
    query: str,
    tenant_id: str = "default",
    top_k: int = config.TOP_K,
    candidate_k: int = config.CANDIDATE_K,
    max_distance: float = config.MAX_DISTANCE,
) -> list[Hit]:
    """
    하이브리드 검색으로 관련 청크를 가져온다.

    벡터 후보 중 임계값 안에 드는 것이 하나도 없으면 빈 목록을 반환한다.
    이 경우 호출한 쪽은 LLM에 아무 문맥도 넣지 말아야 한다. 관련 없는 문맥을
    넣으면 모델이 그럴듯한 거짓말을 만든다.

    Args:
        conn (psycopg.Connection): 연결 객체
        query (str): 사용자 질의문
        tenant_id (str): 테넌트 식별자
        top_k (int): 최종 반환 개수
        candidate_k (int): 검색기별 1차 후보 개수
        max_distance (float): 이 거리보다 가까운 후보가 없으면 포기한다

    Returns:
        list[Hit]: 융합 점수 내림차순 청크 목록. 근거가 없으면 빈 목록
    """
    dense = vector_search(conn, llm.embed_query(query), tenant_id, candidate_k)
    sparse = keyword_search(conn, query, tenant_id, candidate_k)

    hits: dict[str, Hit] = {}

    for chunk_id, doc_id, page, content, distance in dense:
        hits[chunk_id] = Hit(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page=page,
            content=content,
            distance=float(distance),
            found_by={"vector"},
        )

    for rank, (chunk_id, doc_id, page, content, _score) in enumerate(sparse, start=1):
        hit = hits.get(chunk_id)
        if hit is None:
            hit = Hit(chunk_id=chunk_id, doc_id=doc_id, page=page, content=content)
            hits[chunk_id] = hit
        hit.keyword_rank = rank
        hit.found_by.add("keyword")

    # 근거가 있는지 먼저 판단한다. 벡터 거리로만 본다 (키워드는 흔한 단어에도 걸린다)
    distances = [h.distance for h in hits.values() if h.distance is not None]
    if not distances or min(distances) > max_distance:
        return []

    scores = rrf_fuse(
        {
            "vector": [row[0] for row in dense],
            "keyword": [row[0] for row in sparse],
        }
    )
    for chunk_id, score in scores.items():
        hits[chunk_id].score = score

    ranked = sorted(hits.values(), key=lambda h: h.score, reverse=True)
    return ranked[:top_k]


def build_context(
    hits: list[Hit], char_budget: int = config.CONTEXT_CHAR_BUDGET
) -> tuple[str, list[Hit]]:
    """
    검색 결과를 번호가 붙은 문맥 문자열로 조립한다.

    번호는 1부터 시작하며 반환된 목록의 순서와 일치한다. LLM이 [1], [2]로
    인용하면 그대로 출처를 되짚을 수 있다.

    Args:
        hits (list[Hit]): 검색 결과
        char_budget (int): 문맥 전체 길이 상한(문자)

    Returns:
        tuple[str, list[Hit]]: (문맥 문자열, 실제로 담긴 청크 목록)
    """
    blocks: list[str] = []
    used: list[Hit] = []
    total = 0

    for hit in hits:
        block = f"[{len(used) + 1}] (출처: {hit.doc_id}, {hit.page}쪽)\n{hit.content}"
        if used and total + len(block) > char_budget:
            break

        blocks.append(block)
        used.append(hit)
        total += len(block)

    return "\n\n".join(blocks), used


def main() -> None:
    parser = argparse.ArgumentParser(description="하이브리드 검색을 실행한다")
    parser.add_argument("query", nargs="?", default="LLMOps는 무엇인가?")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--tenant", default="default")
    args = parser.parse_args()

    print(f"[질의] {args.query}")
    print(f"[tsquery] {to_tsquery(args.query) or '(없음)'}\n")

    with db.connect() as conn:
        hits = retrieve(conn, args.query, tenant_id=args.tenant, top_k=args.k)

    if not hits:
        print(f"임계값({config.MAX_DISTANCE}) 안에 드는 근거가 없다.")
        return

    for i, hit in enumerate(hits, start=1):
        distance = f"{hit.distance:.4f}" if hit.distance is not None else "-"
        rank = hit.keyword_rank if hit.keyword_rank is not None else "-"
        found = "+".join(sorted(hit.found_by))
        print(f"--- [{i}] {hit.doc_id} {hit.page}쪽 ---")
        print(f"    거리 {distance} / 키워드순위 {rank} / RRF {hit.score:.5f} / {found}")
        print(f"    {hit.content[:200].replace(chr(10), ' ')}\n")


if __name__ == "__main__":
    main()
