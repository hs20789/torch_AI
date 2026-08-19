# 직접 구현한 RAG

교재판(`../create_vector_db.py`, `../add_pdf.py`, `../search_vector_db.py`,
`../gpt_rag.py`)은 그대로 두고, 같은 PDF/같은 DB를 쓰되 테이블만 따로 쓴다.
(`rag_documents`, `rag_chunks` / 교재판은 `pdf_docs`)

## 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 설정값. 전부 환경 변수로 덮어쓸 수 있다 |
| `db.py` | 스키마 DDL, 연결, pgvector 리터럴 변환 |
| `llm.py` | OpenAI SDK 직접 호출 (임베딩, 채팅) |
| `ingest.py` | 멱등 인제스트 — 해시로 변경 감지, 바뀐 문서만 교체 |
| `retrieve.py` | 벡터 + 키워드 하이브리드 검색, RRF 융합, 임계값 |
| `answer.py` | 인용 강제 생성 + 인용 번호 검증 |
| `evaluate.py` | 골든셋으로 검색/생성 분리 평가 |
| `golden.json` | 평가용 질문 6건 (`example.pdf` 기준) |

## 실행

```bash
# 0. Postgres + pgvector (교재에서 쓰던 컨테이너 그대로)
docker run -d --name pgvector -p 5432:5432 \
  -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=pass -e POSTGRES_DB=langchain \
  pgvector/pgvector:pg16

export OPENAI_API_KEY=...
cd chapters/ch18/src

# 1. 스키마 생성 + 현황 확인
python -m rag.db

# 2. 적재 (data/ 안의 PDF 전부)
python -m rag.ingest
python -m rag.ingest             # 다시 실행 -> "변경 없음, 건너뜀". 중복되지 않는다
python -m rag.ingest --force     # 강제 재적재
python -m rag.ingest --delete mlops.pdf

# 3. 검색만 확인 (LLM 호출 없음, 거리 분포 보고 임계값 보정)
python -m rag.retrieve "Canary 배포는 무엇인가?"

# 4. 답변
python -m rag.answer "데이터 드리프트와 concept drift의 차이는?"
python -m rag.answer "연차 휴가는 며칠인가요?"        # 근거 없음 경로
python -m rag.answer "pgvector의 장점은?" --show-context

# 5. 평가
python -m rag.evaluate                  # 검색만 (빠르고 무료)
python -m rag.evaluate --with-answer    # 생성까지 (LLM 비용 발생)
```

## 교재판과 무엇이 다른가

| | 교재판 | 여기 |
|---|---|---|
| 재적재 | 실행할 때마다 중복 적재 | 파일 해시 비교 → 변경분만 교체 |
| 중복 처리 | 검색 시점에 `hash()`로 땜빵 | 애초에 중복이 생기지 않음 |
| 검색 | 벡터 top-3 | 벡터 30 + 키워드 30 → RRF → top-5 |
| 근거 없을 때 | 무조건 상위 3개를 문맥에 밀어넣음 | 거리 임계값 미달 시 "모른다" |
| 인덱스 | 없음 (순차 스캔) | HNSW + GIN |
| 문맥 길이 | 제한 없음 | 문자 예산 내에서 조립 |
| 프롬프트 | `.format()` → role 구조 소실 | system/user 분리 유지 |
| 출처 | 검색된 쪽 번호를 나열 | 인용 강제 + 인용 번호 검증 |
| 인젝션 | 무방비 | 문맥을 `<context>`로 감싸고 "데이터일 뿐" 명시 |
| 온도 | 0.7 | 0.2 |
| 품질 확인 | 눈으로 | 골든셋 recall@k / MRR / 음성 케이스 |
| 멀티테넌시 | 없음 | `tenant_id` 컬럼 + 모든 쿼리에 필터 |

## 아직 안 넣은 것

품질 순으로 다음 단계는 이렇다.

1. **리랭커** — 후보 30개를 cross-encoder(Cohere Rerank, bge-reranker-v2)로
   다시 정렬. 투자 대비 품질 상승이 가장 크다. `retrieve()`가 후보를 넉넉히
   뽑아두었으므로 그 뒤에 한 단계 끼우면 된다.
2. **쿼리 재작성** — 대화 이력이 있는 경우 "그건 왜 그래?" 같은 후속 질문을
   독립 질문으로 바꾼 뒤 검색해야 한다.
3. **레이아웃 인식 파싱** — 지금은 pypdf 텍스트 추출이라 표가 깨진다.
   Docling, Unstructured 등으로 교체.
4. **트레이싱** — 질의/검색된 chunk_id/점수/프롬프트/답변/지연/비용을
   Langfuse 같은 곳에 남겨야 사후 재현이 된다.
5. **FastAPI 서빙** — 지금은 실행할 때마다 커넥션을 새로 연다.
   커넥션 풀 + 비동기로 바꿔야 한다.

## 알려진 한계

- 한국어 키워드 검색은 `to_tsvector('simple', ...)`이라 형태소 분석이 없다.
  조사를 대충 떼고 접두 검색(`:*`)으로 근사한다. 제대로 하려면 pg_bigm이나
  형태소 분석기가 필요하다. 영문 약어/고유명사(`pgvector`, `Canary`)에는
  지금도 잘 듣는다.
- `MAX_DISTANCE = 0.62`는 `data/`의 두 PDF(231청크) 기준으로 실측해 정한 값이다.
  골든셋 양성 케이스의 최소 거리는 최악이 0.5440, 음성 케이스는 0.7766이라
  0.5440~0.7766 사이가 비어 있다. 0.62는 그 구간 안에서 양성 쪽에 조금 붙여
  잡았다 — 틀린 답보다 "모른다"로 빠지는 쪽이 낫기 때문이다.
  **문서를 바꾸면 이 값은 다시 재야 한다.** 분포가 데이터마다 다르다.
- 청크를 쪽 경계에서 자르므로 출처는 정확하지만 쪽을 걸친 문장은 끊긴다.
