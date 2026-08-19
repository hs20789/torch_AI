"""골든셋으로 검색과 생성을 나눠서 평가한다.

이게 교재 코드와 실무 코드의 가장 큰 차이다. 청크 크기, 임베딩 모델,
RRF 상수, 임계값을 바꿨을 때 "좋아졌다"를 감으로 말하면 안 된다.

검색 지표와 생성 지표를 반드시 분리해야 한다. 답이 틀렸을 때
  - 검색이 정답 청크를 못 가져온 것인지
  - 가져왔는데 모델이 못 쓴 것인지
구분되지 않으면 어디를 고쳐야 할지 알 수 없다.

여기서는 "기대하는 문서의 기대하는 쪽이 검색 결과에 들어있는가"를 정답 기준으로 쓴다.
실무에서는 정답 chunk_id를 직접 라벨링해두고 recall@k, MRR, nDCG를 잰다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

from . import answer as answer_module
from . import config, db, retrieve

GOLDEN_PATH = Path(__file__).with_name("golden.json")


def load_cases(path: Path = GOLDEN_PATH) -> list[dict]:
    """골든셋을 읽는다."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval(
    conn: psycopg.Connection, cases: list[dict], top_k: int
) -> tuple[list[dict], list[dict]]:
    """
    검색만 평가한다. LLM을 호출하지 않으므로 빠르고 싸다.

    Args:
        conn (psycopg.Connection): 연결 객체
        cases (list[dict]): 골든셋
        top_k (int): 검색 개수

    Returns:
        tuple[list[dict], list[dict]]: (양성 케이스 결과, 음성 케이스 결과)
    """
    positive, negative = [], []

    for case in cases:
        hits = retrieve.retrieve(conn, case["question"], top_k=top_k)

        if case.get("expect_no_answer"):
            negative.append({"case": case, "passed": not hits, "n_hits": len(hits)})
            continue

        # 쪽 번호만 보면 다른 문서의 같은 쪽이 정답으로 잡힌다.
        # data/에 PDF가 둘 이상이면 doc_id까지 맞아야 정답이다.
        expected_doc = case.get("doc_id")
        expected_pages = set(case["expect_pages"])
        located = [f"{h.doc_id}:{h.page}" for h in hits]
        rank = next(
            (
                i
                for i, hit in enumerate(hits, start=1)
                if hit.page in expected_pages
                and (expected_doc is None or hit.doc_id == expected_doc)
            ),
            None,
        )
        positive.append({"case": case, "hits": hits, "pages": located, "rank": rank})

    return positive, negative


def evaluate_generation(
    conn: psycopg.Connection, cases: list[dict], top_k: int
) -> list[dict]:
    """
    생성까지 평가한다. 케이스마다 LLM을 호출하므로 비용이 든다.

    Args:
        conn (psycopg.Connection): 연결 객체
        cases (list[dict]): 골든셋 (양성 케이스만)
        top_k (int): 문맥 개수

    Returns:
        list[dict]: 케이스별 결과
    """
    results = []

    for case in cases:
        result = answer_module.answer_question(conn, case["question"], top_k=top_k)
        keywords = case.get("answer_keywords", [])
        hit_keywords = [
            kw for kw in keywords if kw.lower() in result.text.lower()
        ]
        results.append(
            {
                "case": case,
                "answer": result,
                "keywords": keywords,
                "hit_keywords": hit_keywords,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="골든셋으로 RAG를 평가한다")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="생성까지 평가한다 (LLM 호출 비용 발생)",
    )
    args = parser.parse_args()

    cases = load_cases()
    positives = [c for c in cases if not c.get("expect_no_answer")]

    with db.connect() as conn:
        found, negative = evaluate_retrieval(conn, cases, args.k)

        print("=" * 60)
        print(f"검색 평가 (top_k={args.k}, 임계값={config.MAX_DISTANCE})")
        print("=" * 60)

        for row in found:
            case, rank = row["case"], row["rank"]
            mark = "PASS" if rank else "FAIL"
            detail = (
                f"{case['expect_pages']}쪽을 {rank}위에서 찾음"
                if rank
                else f"못 찾음 (검색된 쪽: {row['pages']})"
            )
            print(f"  [{mark}] {case['id']:<18} {detail}")

        for row in negative:
            mark = "PASS" if row["passed"] else "FAIL"
            detail = (
                "근거 없음으로 정상 처리"
                if row["passed"]
                else f"근거가 없어야 하는데 {row['n_hits']}건을 반환"
            )
            print(f"  [{mark}] {row['case']['id']:<18} {detail}")

        n_found = sum(1 for r in found if r["rank"])
        mrr = sum(1.0 / r["rank"] for r in found if r["rank"]) / max(len(found), 1)
        n_neg = sum(1 for r in negative if r["passed"])
        print(
            f"\n  recall@{args.k} {n_found}/{len(found)} "
            f"({n_found / max(len(found), 1):.0%}) / MRR {mrr:.3f} / "
            f"음성 {n_neg}/{len(negative)}"
        )

        if not args.with_answer:
            print("\n(생성 평가는 --with-answer)")
            return

        print("\n" + "=" * 60)
        print("생성 평가")
        print("=" * 60)

        results = evaluate_generation(conn, positives, args.k)

    total_tokens = 0
    n_pass = 0

    for row in results:
        result = row["answer"]
        keywords, hit = row["keywords"], row["hit_keywords"]
        ok = len(hit) == len(keywords) and not result.invalid_citations
        n_pass += ok
        total_tokens += result.prompt_tokens + result.completion_tokens

        notes = [f"키워드 {len(hit)}/{len(keywords)}"]
        notes.append(f"인용 {result.cited or '없음'}")
        if result.invalid_citations:
            notes.append(f"가짜 인용 {result.invalid_citations}")

        print(f"  [{'PASS' if ok else 'FAIL'}] {row['case']['id']:<18} {', '.join(notes)}")

    print(f"\n  통과 {n_pass}/{len(results)} / 총 {total_tokens} 토큰")


if __name__ == "__main__":
    main()
