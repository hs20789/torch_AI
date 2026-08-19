"""OpenAI SDK 직접 호출.

교재판은 LangChain의 ChatOpenAI/OpenAIEmbeddings를 썼다. 여기서는 SDK를
그대로 부른다. role 구조가 유지되고, 재시도/타임아웃/토큰 사용량이 전부
눈에 보인다는 게 차이다.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from . import config

_client: OpenAI | None = None


def client() -> OpenAI:
    """
    OpenAI 클라이언트를 반환한다. 프로세스당 한 번만 만든다.

    max_retries는 SDK가 지수 백오프로 처리해준다. 직접 짤 필요 없다.

    Returns:
        OpenAI: 클라이언트 인스턴스
    """
    global _client
    if _client is None:
        _client = OpenAI(max_retries=5, timeout=60.0)
    return _client


def embed_texts(
    texts: list[str],
    model: str = config.EMBEDDING_MODEL,
    batch_size: int = config.EMBEDDING_BATCH,
) -> list[list[float]]:
    """
    텍스트 목록을 배치로 임베딩한다.

    Args:
        texts (list[str]): 임베딩할 텍스트 목록
        model (str): 임베딩 모델 이름
        batch_size (int): 한 요청에 보낼 개수

    Returns:
        list[list[float]]: 입력과 같은 순서의 임베딩 벡터 목록
    """
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        response = client().embeddings.create(model=model, input=chunk)
        # 응답 순서가 보장된다고 가정하지 않고 index로 정렬한다
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)

    return vectors


def embed_query(text: str, model: str = config.EMBEDDING_MODEL) -> list[float]:
    """
    질의문 하나를 임베딩한다.

    Args:
        text (str): 질의문
        model (str): 임베딩 모델 이름

    Returns:
        list[float]: 임베딩 벡터
    """
    return embed_texts([text], model=model)[0]


@dataclass
class ChatResult:
    """LLM 호출 결과와 사용량."""

    text: str
    prompt_tokens: int
    completion_tokens: int


def chat(
    system: str,
    user: str,
    model: str = config.CHAT_MODEL,
    temperature: float = config.CHAT_TEMPERATURE,
) -> ChatResult:
    """
    system/user 역할을 유지한 채로 LLM을 호출한다.

    Args:
        system (str): 시스템 지시문
        user (str): 사용자 메시지
        model (str): 모델 이름
        temperature (float): 생성 온도

    Returns:
        ChatResult: 생성된 텍스트와 토큰 사용량
    """
    response = client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    usage = response.usage
    return ChatResult(
        text=response.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
