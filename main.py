from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import edge_tts
import pymupdf4llm
import chromadb
import httpx
import asyncio
import io
import json
import os
import re
import shutil
import uuid
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel

# ── 앱 초기화 ───────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ChromaDB 초기화 (로컬 영구 저장) ────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="resumes",
    metadata={"hnsw:space": "cosine"},
)

# ── Whisper STT 모델 초기화 ─────────────────────────────────
_WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
whisper_model = WhisperModel(_WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

# ── 직군별 STT initial_prompt ───────────────────────────────
# Whisper가 이 어휘 분포를 기반으로 기술 용어를 우선 인식

_STT_PROMPTS_KO: dict[str, str] = {
    "backend": (
        "저는 백엔드 개발자로 자바, 스프링 부트, 파이썬, 장고를 사용합니다. "
        "REST API, JPA, 하이버네이트, 마이크로서비스 아키텍처, MSA를 경험했습니다. "
        "MySQL, PostgreSQL, Redis, MongoDB 등 데이터베이스를 다루며, "
        "도커, 쿠버네티스, CI/CD, AWS 클라우드에 배포한 경험이 있습니다. "
        "알고리즘, 자료구조, 객체지향 프로그래밍, 디자인 패턴, 트랜잭션, 인덱스를 공부했습니다."
    ),
    "frontend": (
        "저는 프론트엔드 개발자로 리액트, 뷰, 앵귤러, 타입스크립트를 사용합니다. "
        "HTML, CSS, 자바스크립트, 웹팩, 바이트, 상태 관리 라이브러리를 다룹니다. "
        "리덕스, 리코일, 주스탠드 같은 상태 관리와 리액트 쿼리를 활용했습니다. "
        "웹 접근성, 반응형 디자인, 크로스 브라우징, 성능 최적화, SEO를 고려한 개발을 합니다. "
        "REST API 연동, GraphQL, 웹소켓, UI/UX 설계 경험이 있습니다."
    ),
    "fullstack": (
        "저는 풀스택 개발자로 프론트엔드와 백엔드 모두 개발합니다. "
        "리액트, 타입스크립트, 노드제이에스, 자바, 스프링 부트를 사용합니다. "
        "REST API, 데이터베이스 설계, 클라우드 배포, CI/CD 파이프라인을 구축했습니다. "
        "MySQL, MongoDB, Redis, 도커, 쿠버네티스, AWS를 경험했습니다."
    ),
    "data": (
        "저는 데이터 엔지니어 또는 데이터 사이언티스트입니다. "
        "파이썬, 판다스, 넘파이, 사이킷런, 텐서플로우, 파이토치를 사용합니다. "
        "머신러닝, 딥러닝, 자연어 처리, 컴퓨터 비전, 강화학습을 공부했습니다. "
        "SQL, 스파크, 하둡, 카프카, 에어플로우, 데이터 파이프라인을 다루며, "
        "A/B 테스트, 피처 엔지니어링, 모델 튜닝, MLOps 경험이 있습니다."
    ),
    "devops": (
        "저는 DevOps 엔지니어로 인프라와 배포 자동화를 담당합니다. "
        "도커, 쿠버네티스, 헬름, 테라폼, 앤서블을 사용합니다. "
        "AWS, GCP, Azure 클라우드 인프라를 설계하고 운영합니다. "
        "CI/CD 파이프라인, 깃허브 액션, 젠킨스, 아르고CD를 구축했습니다. "
        "프로메테우스, 그라파나, ELK 스택으로 모니터링과 로깅을 구성했습니다."
    ),
    "mobile": (
        "저는 모바일 개발자로 안드로이드와 iOS 앱을 개발합니다. "
        "코틀린, 자바, 스위프트, 플러터, 리액트 네이티브를 사용합니다. "
        "젯팩 컴포즈, SwiftUI, 상태 관리, 비동기 처리를 경험했습니다. "
        "REST API 연동, 로컬 데이터베이스, 푸시 알림, 앱 성능 최적화를 다룹니다. "
        "구글 플레이스토어, 앱스토어 배포 경험이 있습니다."
    ),
    "security": (
        "저는 보안 엔지니어로 취약점 분석과 침투 테스트를 수행합니다. "
        "웹 취약점, OWASP Top 10, SQL 인젝션, XSS, CSRF를 분석합니다. "
        "암호화, 해시, 공개키 기반 구조, TLS, 인증과 인가를 이해합니다. "
        "버프 스위트, 메타스플로잇, 와이어샤크, 네트워크 패킷 분석을 활용합니다. "
        "보안 정책, 컴플라이언스, 취약점 보고서 작성 경험이 있습니다."
    ),
    "game": (
        "저는 게임 개발자로 유니티와 언리얼 엔진을 사용합니다. "
        "C샵, C 플플, 게임 오브젝트, 컴포넌트 패턴, 렌더링 파이프라인을 공부했습니다. "
        "물리 엔진, 충돌 처리, 애니메이션, 셰이더, 최적화 기법을 경험했습니다. "
        "멀티플레이어 네트워크, 게임 서버, 매치메이킹 시스템을 다룹니다."
    ),
    "default": (
        "저는 소프트웨어 개발자입니다. "
        "알고리즘, 자료구조, 객체지향 프로그래밍, 디자인 패턴을 공부했습니다. "
        "자바, 파이썬, 자바스크립트, 타입스크립트, C 플플 중 하나 이상을 사용합니다. "
        "데이터베이스, REST API, 깃허브, 도커, 클라우드 환경을 경험했습니다. "
        "트러블슈팅, 리팩토링, 코드 리뷰, 애자일 방법론에 익숙합니다."
    ),
}

_STT_PROMPTS_EN: dict[str, str] = {
    "backend": (
        "I'm a backend developer using Java, Spring Boot, Python, Django, Node.js. "
        "REST APIs, JPA, Hibernate, microservices, MySQL, PostgreSQL, Redis, MongoDB. "
        "Docker, Kubernetes, CI/CD, AWS, algorithms, data structures, design patterns."
    ),
    "frontend": (
        "I'm a frontend developer using React, Vue, Angular, TypeScript, JavaScript. "
        "Webpack, Vite, Redux, Recoil, React Query, HTML, CSS, responsive design, SEO."
    ),
    "data": (
        "I'm a data scientist using Python, Pandas, NumPy, scikit-learn, TensorFlow, PyTorch. "
        "Machine learning, deep learning, NLP, computer vision, SQL, Spark, Kafka, MLOps."
    ),
    "devops": (
        "I'm a DevOps engineer using Docker, Kubernetes, Helm, Terraform, Ansible. "
        "AWS, GCP, Azure, CI/CD, GitHub Actions, Jenkins, ArgoCD, Prometheus, Grafana."
    ),
    "default": (
        "I'm a software engineer. Algorithms, data structures, object-oriented programming, design patterns. "
        "Java, Python, JavaScript, TypeScript, databases, REST APIs, Git, Docker, cloud deployment."
    ),
}

# 직군 키워드 → 프롬프트 키 매핑
_JOB_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["프론트엔드", "frontend", "front-end", "프론트", "퍼블리셔", "ui", "ux"], "frontend"),
    (["데이터", "data", "ml", "ai", "머신러닝", "딥러닝", "분석", "사이언티스트", "scientist", "엔지니어링"], "data"),
    (["devops", "데브옵스", "인프라", "infra", "클라우드", "cloud", "sre", "운영"], "devops"),
    (["모바일", "mobile", "android", "안드로이드", "ios", "flutter", "플러터"], "mobile"),
    (["보안", "security", "침투", "취약점", "해킹"], "security"),
    (["게임", "game", "unity", "유니티", "unreal", "언리얼"], "game"),
    (["풀스택", "fullstack", "full-stack", "full stack"], "fullstack"),
    (["백엔드", "backend", "back-end", "서버", "server"], "backend"),
]

def _resolve_job_keys(desired_job: str) -> list[str]:
    """desired_job 문자열에서 매칭되는 모든 프롬프트 키를 반환."""
    if not desired_job:
        return ["default"]
    lower = desired_job.lower()
    matched = [key for keywords, key in _JOB_KEYWORD_MAP if any(kw in lower for kw in keywords)]
    return matched if matched else ["default"]

def _get_stt_prompt(desired_job: str, language: str) -> str:
    keys = _resolve_job_keys(desired_job)
    prompts = _STT_PROMPTS_EN if language == "en" else _STT_PROMPTS_KO
    if len(keys) == 1:
        return prompts.get(keys[0], prompts["default"])
    # 여러 직군 매칭 시: 각 프롬프트의 첫 문장만 결합
    # (Whisper initial_prompt는 어휘 프라이밍 목적 — 짧게 유지)
    parts = []
    for key in keys:
        full = prompts.get(key, "")
        first_sentence = full.split(". ")[0] + "." if full else ""
        if first_sentence:
            parts.append(first_sentence)
    return " ".join(parts) if parts else prompts["default"]

# STT 후처리 보정 사전 (발음 유사어 → 정규 표기)
_STT_CORRECTIONS_KO = {
    # 프레임워크 / 언어
    "리엑트": "리액트",
    "자바스클립트": "자바스크립트",
    "자바 스크립트": "자바스크립트",
    "타입스클립트": "타입스크립트",
    "타입 스크립트": "타입스크립트",
    "파이선": "파이썬",
    "파이슨": "파이썬",
    "코틀린": "코틀린",
    "스프링부트": "스프링 부트",
    # 인프라 / 도구
    "쿠버네티즈": "쿠버네티스",
    "쿠버네이티스": "쿠버네티스",
    "쿠버네이티즈": "쿠버네티스",
    "도클": "도커",
    "깃헙": "깃허브",
    "깃합": "깃허브",
    "게더허브": "깃허브",
    "포스트그래스": "포스트그레스",
    "레디": "레디스",
    "몽고": "몽고디비",
    "카프": "카프카",
    # CS 개념
    "알로리즘": "알고리즘",
    "알고리즘": "알고리즘",  # 정규화 (중복 방지)
    "자료구": "자료구조",
    "오브젝 지향": "객체지향",
    "오브젝트지향": "객체지향",
    "디자인패턴": "디자인 패턴",
    "마이크로 서비스": "마이크로서비스",
    "시아이씨디": "CI/CD",
    "에이피아이": "API",
    "레스트에이피아이": "REST API",
    "오버라이딩": "오버라이딩",
    "오버로딩": "오버로딩",
    "싱글턴": "싱글톤",
    "개비지컬렉션": "가비지 컬렉션",
    "게비지컬렉션": "가비지 컬렉션",
    "트랜젝션": "트랜잭션",
    "트렌젝션": "트랜잭션",
    "인덱싱": "인덱싱",
    "쿼리": "쿼리",
    "컨테이너": "컨테이너",
    "로드밸런싱": "로드 밸런싱",
    "캐슁": "캐싱",
    "리팩토링": "리팩토링",
}

def correct_stt_text(text: str) -> str:
    """STT 결과의 흔한 발음 오인식을 보정."""
    for wrong, correct in _STT_CORRECTIONS_KO.items():
        text = text.replace(wrong, correct)
    return text

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://127.0.0.1:11434")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "nomic-embed-text")
LLM_MODEL  = os.getenv("LLM_MODEL",  "exaone3.5")
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50

# ── 면접관 시스템 프롬프트 ────────────────────────────────────
BASIC_SYSTEM = (
    "당신은 50대 여성 임원이자 23년 차 베테랑 면접관 '박부장'입니다. "
    "인성, 조직 적합성, 갈등 해결 능력, 스트레스 관리 등을 평가합니다. "
    "뻔한 질문은 피하고, 지원자의 이전 답변과 이력서를 깊이 분석하여 구체적인 경험을 묻는 날카로운 꼬리 질문을 던지세요. "
    "매번 다른 인성 주제(협업, 리더십, 실패 경험 등)로 넘어가며 질문을 다채롭게 하세요. "
    "질문은 한 번에 하나씩, 한국어로 하세요. 말투는 젠틀하고 여유로운 임원의 톤을 유지하세요. "
    "대화가 3~4번 오가서 충분히 평가가 되었다고 판단되면, 따뜻한 격려 인사와 함께 반드시 텍스트 끝에 [면접 종료]를 적으세요."
)

JOB_SYSTEM = (
    "당신은 30대 중반 남성이자 17년 차 수석 개발자이며 실무 중심의 깐깐한 면접관 '개발팀 김 팀장'입니다. "
    "직무 역량, 기술적 문제 해결력, 코드 최적화, 아키텍처 이해도를 깊게 파고듭니다. "
    "지원자의 답변과 이력서에서 기술적인 허점이나 더 파고들 부분을 찾아내어 실무 상황을 가정한 압박 질문을 던지세요. "
    "질문은 모두 한국어로 진행하세요."
    "\n\n[면접 진행 규칙]\n"
    "1. 본 질문은 총 5개로 구성하며, 질문 번호를 '[질문 1]', '[질문 2]' 형식으로 앞에 표기하세요.\n"
    "2. 질문 1→5로 갈수록 난이도가 점진적으로 높아지도록 설계하세요. "
    "초반(1~2번)은 기본 개념과 경험, 중반(3~4번)은 트러블슈팅·성능 개선, 후반(5번)은 아키텍처 설계 및 심화 주제를 다루세요.\n"
    "3. 지원자의 답변이 흥미롭거나 더 파고들 부분이 있으면 꼬리 질문을 하세요. "
    "꼬리 질문은 '[질문 2-1]', '[질문 2-2]' 형식으로 표기하며 최소 2회 이상 실시하세요. "
    "꼬리 질문은 총 5개 본 질문에 포함되지 않습니다.\n"
    "4. 모든 말투는 반드시 공손하고 격식 있는 존댓말을 사용하세요. 반말이나 명령조는 절대 사용하지 마세요.\n"
    "5. 질문은 한 번에 하나씩만 하세요.\n"
    "6. 5개의 본 질문과 최소 2개의 꼬리 질문이 모두 완료되면, 짧고 격식 있는 마무리 인사와 함께 반드시 텍스트 끝에 [면접 종료]를 적으세요."
)


# ── 유틸 함수 ────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """단락 우선, 길면 CHUNK_SIZE 기준으로 분할."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= CHUNK_SIZE:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # 단락 자체가 CHUNK_SIZE 초과 시 강제 분할
            while len(para) > CHUNK_SIZE:
                chunks.append(para[:CHUNK_SIZE])
                para = para[CHUNK_SIZE - CHUNK_OVERLAP:]
            current = para
    if current:
        chunks.append(current)
    return chunks


async def get_embedding(text: str) -> list[float]:
    """Ollama 임베딩 API 호출."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/embed 응답: {"embeddings": [[...]] }
        return data["embeddings"][0]


def _get_persona(interview_type: str) -> str:
    """면접 유형에 따른 페르소나 문자열 반환."""
    if interview_type == "job":
        return "17년 차 수석 개발자 면접관 '개발팀 김 팀장'"
    return "23년 차 임원 면접관 '박부장'"


def _extract_json(raw: str) -> dict:
    """LLM 응답에서 JSON 객체를 추출해 파싱. 실패 시 빈 dict 반환."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    return json.loads(raw[start:end])


async def _embed_and_store(chunks: list[str], user_id: str, filename: str) -> None:
    """청크 임베딩 후 ChromaDB에 저장 (기존 user_id 데이터 먼저 삭제)."""
    try:
        # 기존 이력서 청크 삭제
        existing = collection.get(where={"user_id": user_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        embeddings = [await get_embedding(chunk) for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"user_id": user_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩/저장 실패: {e}")


# ── 스키마 ───────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    top_k: int = 5


class EmbedTextRequest(BaseModel):
    user_id: str
    text: str
    filename: str = "resume"


class GeneratePoolRequest(BaseModel):
    user_id: str
    interview_type: str          # "basic" | "job"




class ShouldFollowupRequest(BaseModel):
    interview_type: str
    question_text: str
    user_answer: str
    current_followup_count: int  # 2 이상이면 즉시 false 반환


class GenerateFollowupRequest(BaseModel):
    interview_type: str
    parent_question: str
    user_answer: str




class EvaluateAnswerRequest(BaseModel):
    interview_type: str
    question_text: str
    user_answer: str


class FeedbackQuestion(BaseModel):
    question_id: int
    question_text: str
    answer_text: str

class GenerateFeedbackRequest(BaseModel):
    interview_type: str
    language: str = "ko"
    questions: list[FeedbackQuestion]


# ── 엔드포인트 ───────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/embed-text")
async def embed_text(req: EmbedTextRequest):
    """마크다운 텍스트 → 청크 → 임베딩 → ChromaDB 저장."""
    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="텍스트가 비어 있습니다.")

    await _embed_and_store(chunks, req.user_id, req.filename)
    return {"user_id": req.user_id, "chunks_stored": len(chunks)}


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    """PDF → Markdown 변환만 수행 (저장 없음)."""
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        md_text = pymupdf4llm.to_markdown(temp_path)
        os.remove(temp_path)
        return {"filename": file.filename, "markdown": md_text}
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed-resume")
async def embed_resume(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    """PDF → Markdown → 청크 → 임베딩 → ChromaDB 저장."""
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. PDF → Markdown
        md_text = pymupdf4llm.to_markdown(temp_path)
        os.remove(temp_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"PDF 파싱 실패: {e}")

    # 2. 청크 분할
    chunks = chunk_text(md_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="파싱된 텍스트가 없습니다.")

    # 3. 임베딩 + ChromaDB 저장
    await _embed_and_store(chunks, user_id, file.filename)

    return {
        "filename": file.filename,
        "user_id": user_id,
        "chunks_stored": len(chunks),
        "markdown": md_text,
    }


@app.post("/search")
async def search(req: SearchRequest):
    """쿼리 → 임베딩 → ChromaDB 유사도 검색."""
    try:
        query_emb = await get_embedding(req.query)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다.")

    where = {"user_id": req.user_id} if req.user_id else None

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=req.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "score": round(1 - dist, 4)})

    return {"query": req.query, "results": hits}


async def retrieve_resume_context(user_id: str, query: str, top_k: int = 4) -> str:
    """ChromaDB에서 이력서 관련 청크를 검색해 컨텍스트 문자열로 반환."""
    try:
        query_emb = await get_embedding(query)
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "distances"],
        )
        docs = results["documents"][0]
        dists = results["distances"][0]
        # 유사도 0.3 미만(너무 관련 없는) 청크 필터링
        relevant = [doc for doc, dist in zip(docs, dists) if (1 - dist) >= 0.3]
        return "\n\n".join(relevant) if relevant else ""
    except Exception:
        return ""


async def call_llm(messages: list, timeout: int = 180) -> str:
    """LLM(Ollama) 호출 공통 함수."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": LLM_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {type(e).__name__}: {e}")


@app.post("/generate-pool")
async def generate_pool(req: GeneratePoolRequest):
    """
    면접 시작 전 신규 질문 5개 생성 (카테고리 A 3개 + B 2개).
    질문 1개씩 개별 호출하여 토큰 깨짐 방지.
    """
    resume_context = await retrieve_resume_context(req.user_id, "프로젝트 기술스택 경험")
    if not resume_context:
        raise HTTPException(status_code=400, detail="이력서 정보를 찾을 수 없습니다. 마이페이지에서 이력서를 먼저 업로드해 주세요.")

    system_msg = "You are a Korean interviewer. You MUST respond only in Korean (한국어). Never use English, Japanese, Chinese, or any other language. Use correct Korean spelling and spacing."

    persona = _get_persona(req.interview_type)

    import random

    if req.interview_type == "job":
        # 신입 수준 CS 기초 주제 풀 (개념 설명 가능한 수준)
        cs_topics_pool = [
            "HTTP 메서드(GET/POST/PUT/DELETE)의 차이",
            "HTTP와 HTTPS의 차이",
            "프로세스와 스레드의 차이",
            "RDB와 NoSQL의 차이와 사용 사례",
            "DB 인덱스의 역할",
            "TCP와 UDP의 차이",
            "가비지 컬렉션(GC) 개념",
            "RESTful API란 무엇인가",
            "캐시의 역할과 필요성",
            "동기와 비동기 처리 방식의 차이",
            "트랜잭션과 ACID 속성",
            "쿠키와 세션의 차이",
            "JWT 인증 방식 개념",
            "CI/CD란 무엇이며 왜 사용하는가",
            "스택과 큐의 차이와 사용 예시",
            "객체지향 프로그래밍의 4가지 특징",
            "CORS란 무엇이며 왜 발생하는가",
            "SQL의 JOIN 종류와 차이",
            "웹 브라우저 주소창에 URL 입력 시 일어나는 일",
            "MVC 패턴이란 무엇인가",
        ]
        selected_topics = random.sample(cs_topics_pool, 3)
        category_specs = [
            ("A", f"CS 기초 개념 질문 — 주제: [{selected_topics[0]}]. 개념을 말로 설명하는 수준. 코드 구현 요구 절대 금지. 신입 지원자가 답할 수 있는 쉬운 난이도."),
            ("A", f"CS 기초 개념 질문 — 주제: [{selected_topics[1]}]. 개념 설명 수준. 코드 구현 요구 절대 금지. 신입 지원자가 답할 수 있는 쉬운 난이도."),
            ("A", f"CS 기초 개념 질문 — 주제: [{selected_topics[2]}]. 개념 설명 수준. 코드 구현 요구 절대 금지. 신입 지원자가 답할 수 있는 쉬운 난이도."),
            ("B", "프로젝트/기술 역량 질문 — 이력서의 특정 프로젝트 기능을 어떤 방식으로 구현했는지·왜 그 방식을 선택했는지, 또는 현재 관심 있는 기술이나 최근 공부한 내용을 묻는 질문"),
            ("B", "프로젝트 심화 질문 — 이력서 프로젝트에서 가장 도전적이었던 부분과 해결 방법, 또는 팀 프로젝트 실패 경험과 교훈, 또는 버그 발견·테스트 프로세스를 묻는 질문. 앞 질문과 다른 주제."),
        ]
    else:
        personality_topics_pool = [
            "개발자로서 5년/10년 후 목표",
            "개발이 적성에 맞는 이유",
            "개발자에게 가장 중요한 역량",
            "주니어 개발자로서 가장 먼저 키우고 싶은 역량",
            "압박감이 클 때 일하는 방식",
            "본인의 강점이 직무와 맞는 이유",
            "본인의 강점과 약점",
            "실패 경험과 교훈",
            "개발 외 자기계발 방법",
            "최근 가장 흥미롭게 공부한 기술 주제",
            "개발 프로젝트에서 가장 보람 있었던 순간",
            "코딩을 시작하게 된 계기",
        ]
        teamwork_topics_pool = [
            "팀 내 갈등 해결 경험",
            "의견 충돌 시 팀원 설득 방법",
            "팀 프로젝트에서 본인의 역할",
            "효과적인 팀 커뮤니케이션 방법",
            "업무 스타일이 다른 팀원과 협업 경험",
            "팀 프로젝트에서 협력해 문제를 해결한 경험",
            "스트레스·압박 상황 대처 방식",
            "팀 프로젝트에서 일정이 밀렸을 때 대응 방법",
        ]
        selected_personality = random.sample(personality_topics_pool, 3)
        selected_teamwork = random.sample(teamwork_topics_pool, 2)
        category_specs = [
            ("A", f"인성 질문 — 주제: [{selected_personality[0]}]. 지원자의 가치관과 성장 의지를 파악하는 질문."),
            ("A", f"인성 질문 — 주제: [{selected_personality[1]}]."),
            ("A", f"이력서/경험 기반 인성 질문 — 주제: [{selected_personality[2]}]. 지원자의 실제 경험을 기반으로."),
            ("B", f"팀워크/소프트스킬 질문 — 주제: [{selected_teamwork[0]}]."),
            ("B", f"팀워크/소프트스킬 질문 — 주제: [{selected_teamwork[1]}]."),
        ]

    resume_section = ""
    if resume_context:
        resume_section = f"\n[지원자 이력서 발췌 - 참고용]\n{resume_context}\n"

    rules = (
        "규칙: "
        "1) 질문은 반드시 1문장, 1개의 질문만. 설명·배경·힌트를 앞에 붙이지 말 것. 두 개의 질문을 이어 붙이는 것 절대 금지. "
        "1-1) 올바른 예: 'HTTP 메서드 GET과 POST의 차이를 설명해 주세요.' "
        "1-2) 잘못된 예(절대 금지): 'GET은 데이터를 조회할 때 사용합니다. 각각의 차이를 설명해 주세요.' (앞에 설명 붙이는 것 금지) "
        "1-3) 잘못된 예(절대 금지): 'ChromaDB 활용 방식을 설명해 주세요. 왜 이 방법을 선택하셨나요?' (두 질문 이어 붙이는 것 금지) "
        "2) ArrayList, Spring Boot 등 고유 기술 용어는 영어 가능. "
        "3) 나머지는 모두 한국어. "
        "4) 질문은 반드시 하나의 어미로만 끝낼 것. 어미가 두 개 이상이면 절대 안 됨. "
        "4-1) 올바른 예: '트랜잭션이란 무엇인가요?' 또는 '동기와 비동기의 차이를 설명해 주세요.' "
        "4-2) 잘못된 예(절대 금지): '트랜잭션이란 무엇인가요? 말씀해 주세요.' 또는 '차이를 설명해 주세요? 무엇인가요?' "
        "4-3) 질문이 인가요/무엇인가요/있으신가요 중 하나로 끝났으면 그 뒤에 아무것도 붙이지 말 것. "
        "5) 띄어쓰기 정확히. "
        "6) 질문 텍스트만 출력하고 다른 말은 절대 하지 말 것. "
        "7) 코드 작성·구현을 요구하는 질문은 절대 생성하지 말 것."
    )

    def is_corrupted(text: str) -> bool:
        return bool(re.search(r'[가-힣][a-zA-Z]{1,}[가-힣]', text))

    async def generate_one(category: str, spec: str) -> dict:
        prompt = (
            f"당신은 {persona}입니다.{resume_section}\n"
            f"다음 유형의 면접 질문 1개를 생성하세요: {spec}\n"
            f"{rules}"
        )
        for _ in range(3):
            raw = await call_llm([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ], timeout=180)
            text = raw.strip().strip('"').strip()
            if text and not is_corrupted(text):
                return {"category": category, "text": text}
        return None  # 3회 실패 시 None

    questions = []
    for category, spec in category_specs:
        result = await generate_one(category, spec)
        questions.append(result)

    # 실패한 항목은 fallback으로 대체
    job_fallback = [
        {"category": "A", "text": "HTTP 메서드(GET, POST, PUT, DELETE)의 차이점을 설명해 주세요."},
        {"category": "A", "text": "프로세스와 스레드의 차이점을 설명해 주세요."},
        {"category": "A", "text": "RDB와 NoSQL의 차이점과 각각 어떤 상황에 적합한지 말씀해 주세요."},
        {"category": "B", "text": "가장 자신 있는 프로젝트에서 본인의 역할과 기여를 말씀해 주세요."},
        {"category": "B", "text": "프로젝트에서 사용한 기술 스택을 선택한 이유를 말씀해 주세요."},
    ]
    basic_fallback = [
        {"category": "A", "text": "개발자로서 5년 후 목표가 있다면 말씀해 주세요."},
        {"category": "A", "text": "본인의 강점이 이 직무와 어떻게 연결된다고 생각하시나요?"},
        {"category": "A", "text": "개발하면서 가장 크게 성장했다고 느낀 경험을 말씀해 주세요."},
        {"category": "B", "text": "팀 프로젝트에서 의견 충돌이 생겼을 때 어떻게 해결하셨는지 말씀해 주세요."},
        {"category": "B", "text": "업무 스타일이 다른 팀원과 함께 일해야 했던 경험과 그 대처 방법을 말씀해 주세요."},
    ]
    fallback = job_fallback if req.interview_type == "job" else basic_fallback

    final = []
    for i, q in enumerate(questions):
        final.append(q if q is not None else fallback[i])

    return {"questions": final}


@app.post("/should-followup")
async def should_followup(req: ShouldFollowupRequest):
    """꼬리 질문 여부 판단. current_followup_count >= 2이면 즉시 false 반환."""
    if req.current_followup_count >= 2:
        return {"should_followup": False}

    prompt = (
        "다음 면접 답변을 읽고, 꼬리 질문이 필요한지 판단하세요.\n\n"
        f"질문: {req.question_text}\n"
        f"답변: {req.user_answer}\n\n"
        "꼬리 질문이 필요한 경우: 답변이 모호하거나, 구체적 사례가 없거나, 기술적으로 불완전할 때\n"
        "꼬리 질문이 불필요한 경우: 답변이 충분히 구체적이고 완결된 경우\n\n"
        '반드시 JSON으로만 응답하세요: {"should_followup": true} 또는 {"should_followup": false}'
    )

    raw = await call_llm([{"role": "user", "content": prompt}])
    try:
        parsed = _extract_json(raw)
        return {"should_followup": bool(parsed.get("should_followup", False))}
    except Exception:
        return {"should_followup": False}


@app.post("/generate-followup")
async def generate_followup(req: GenerateFollowupRequest):
    """꼬리 질문 생성 (부모 질문 + 사용자 답변 컨텍스트)."""
    prompt = (
        f"당신은 {_get_persona(req.interview_type)}입니다.\n"
        f"면접관 질문: {req.parent_question}\n"
        f"지원자 답변: {req.user_answer}\n\n"
        "위 답변 중 가장 불명확하거나 짧게 언급된 부분 한 가지에 대해 간단한 꼬리 질문 1개를 작성하세요. "
        "질문은 짧고 간결하게 한 문장으로, 존댓말로 작성하세요. "
        "질문 텍스트만 출력하고 다른 내용은 절대 포함하지 마세요. "
        "반드시 한국어로만 작성하고, 영어 번역이나 설명을 절대 포함하지 마세요."
    )

    followup = await call_llm([
        {"role": "system", "content": "You are a Korean interviewer. You MUST respond only in Korean (한국어). Never use English, Japanese, Chinese, or any other language. Use correct Korean spelling and spacing."},
        {"role": "user", "content": prompt},
    ])
    return {"followup_question": followup.strip()}


@app.post("/evaluate-answer")
async def evaluate_answer(req: EvaluateAnswerRequest):
    """
    답변 품질 평가 → "GOOD" | "POOR" 반환.
    평가 기준: 답변의 구체성, 완결성, 질문 관련성.
    """
    prompt = (
        "다음 면접 질문과 답변을 평가하세요.\n\n"
        f"질문: {req.question_text}\n"
        f"답변: {req.user_answer}\n\n"
        "평가 기준:\n"
        "- 답변이 질문에 관련된 내용인가?\n"
        "- 구체적인 사례나 근거가 포함되어 있는가?\n"
        "- 답변이 충분히 완결되어 있는가?\n\n"
        "답변이 불충분하거나 모호하면 should_retry: true, 충분하면 should_retry: false.\n"
        '반드시 JSON으로만 응답하세요: {"should_retry": true} 또는 {"should_retry": false}'
    )

    raw = await call_llm([{"role": "user", "content": prompt}])
    try:
        parsed = _extract_json(raw)
        quality = "POOR" if parsed.get("should_retry", False) else "GOOD"
    except Exception:
        quality = "GOOD"

    return {"quality": quality}


# 소프트스킬 채점 루브릭 — 역량별 점수 구간 기준
SKILL_RUBRICS = {
    "job": {
        "기술깊이":    "90+: 개념·원리 명확, 실사용 경험을 구체적 수치·사례로 뒷받침, 한계/트레이드오프 인지 | 70~89: 개념 정확하나 실사용 경험 추상적 | 50~69: 부분 이해, 오개념 혼재 | ~49: 핵심 벗어남·암기 수준·한 단어·무응답",
        "문제해결력":  "90+: 문제 정의→원인 분석→해결 과정→결과를 단계별로 명확히 서술 | 70~89: 해결 과정 있으나 근거·결과 불명확 | 50~69: 해결책만 언급, 과정 생략 | ~49: 문제 인식 부족·무응답",
        "커뮤니케이션":"90+: 두괄식·STAR 구조, 핵심만 간결히 전달, 면접관 이해 고려 | 70~89: 내용 있으나 구조 산만·장황 | 50~69: 표현 모호, 의도 전달 부족 | ~49: 질문과 무관·한 단어 수준",
        "논리적사고":  "90+: 주장에 근거 명확, 반례·예외 인지, 일관된 논리 | 70~89: 논리 흐름 있으나 근거 약하거나 비약 존재 | 50~69: 결론만 있고 논리 과정 생략 | ~49: 논리 구조 없음·무응답",
        "성장가능성":  "90+: 구체적 학습 계획·실행 경험 제시, 자기인식 명확 | 70~89: 성장 의지는 있으나 구체성 부족 | 50~69: 일반적 언급에 그침 | ~49: 성장 의지 없음·무응답",
    },
    "basic": {
        "커뮤니케이션":"90+: 두괄식·STAR 구조, 핵심만 간결히 전달, 상대 배려 표현 | 70~89: 내용 있으나 구조 산만·장황 | 50~69: 표현 모호, 의도 전달 부족 | ~49: 질문과 무관·한 단어 수준",
        "조직적합성":  "90+: 팀·조직 가치관 부합 사례를 구체적으로 제시 | 70~89: 부합 의지 표현되나 사례 부족 | 50~69: 일반적 언급에 그침 | ~49: 관련 내용 없음·무응답",
        "문제해결력":  "90+: 문제 정의→원인 분석→해결 과정→결과 명확 | 70~89: 해결 과정 있으나 결과 불명확 | 50~69: 해결책만 언급, 과정 생략 | ~49: 문제 인식 부족·무응답",
        "리더십":      "90+: 구체적 리더십 발휘 경험, 팀원 동기부여·갈등 해결 사례 | 70~89: 리더십 경험 있으나 영향력 불명확 | 50~69: 역할만 언급, 리더십 내용 부족 | ~49: 관련 경험 없음·무응답",
        "성장가능성":  "90+: 구체적 학습 계획·실행 경험, 자기인식 명확 | 70~89: 성장 의지는 있으나 구체성 부족 | 50~69: 일반적 언급에 그침 | ~49: 성장 의지 없음·무응답",
    },
}


@app.post("/generate-feedback")
async def generate_feedback(req: GenerateFeedbackRequest):
    """
    면접 종합 피드백 생성.
    - 면접 총평 (summary)
    - 소프트스킬 카테고리별 루브릭 기반 채점 (score + evidence + weakness)
    - 질문별: 질문 의도, 답변 피드백, 개선된 답변
    """
    interviewer = "개발팀 김 팀장 (17년 차 수석 개발자)" if req.interview_type == "job" else "박부장 (23년 차 임원)"
    skill_keys = list(SKILL_RUBRICS.get(req.interview_type, SKILL_RUBRICS["job"]).keys())
    rubrics = SKILL_RUBRICS.get(req.interview_type, SKILL_RUBRICS["job"])

    questions_text = "\n\n".join(
        f"[질문 {q.question_id}]\n질문: {q.question_text}\n답변: {q.answer_text}"
        for q in req.questions
    )

    rubric_guide = "\n".join(f"  · {skill}: {desc}" for skill, desc in rubrics.items())

    skill_schema_entries = "\n".join(
        f'    "{c}": {{"score": 정수(0~100), "evidence": "이 점수를 매긴 근거 (답변 내용 직접 인용)", "weakness": "이 역량에서 부족한 점 (없으면 빈 문자열)"}}'
        for c in skill_keys
    )

    question_schema_lines = []
    for q in req.questions:
        question_schema_lines.append(
            '    {"question_id": ' + str(q.question_id) +
            ', "answer_summary": "지원자 답변을 1~2문장으로 요약 (핵심 내용만, 없거나 무의미한 답변이면 \'답변 없음\')", "intent": "이 질문의 의도", "feedback": "이 답변에 대한 구체적 피드백", "improved_answer": "90점 이상의 모범 답변을 1인칭으로 직접 작성 (메타 설명 금지, 실제 답변 문장으로 작성)"}'
        )
    question_schema = ",\n".join(question_schema_lines)

    prompt = (
        f"당신은 면접관 '{interviewer}'입니다.\n"
        f"아래는 면접에서 오간 질문과 실제 지원자의 답변입니다.\n\n"
        f"{questions_text}\n\n"
        "【중요 평가 원칙】\n"
        "- 반드시 지원자가 실제로 작성한 답변 내용만을 기준으로 평가하세요.\n"
        "- 답변이 짧거나, 의미 없거나(예: 'sss', '모름', 한 단어), 질문과 무관하다면 그렇다고 솔직하게 평가하세요.\n"
        "- 실제 답변에 없는 내용을 '잘 설명했다'거나 '명확하게 전달했다'는 식으로 긍정 평가하지 마세요.\n"
        "- feedback과 evidence는 실제 답변의 구체적인 내용을 인용하거나 언급하며 작성하세요.\n\n"
        "【improved_answer 작성 원칙】\n"
        "- improved_answer는 해당 질문에 대한 '90점 이상의 모범 답변'을 직접 작성하세요.\n"
        "- 실제 면접에서 말하듯 자연스럽고 구체적인 1인칭 문장으로 작성하세요 (예: '저는 ~했습니다. ~를 통해 ~를 해결했습니다.').\n"
        "- 지원자의 실제 답변에서 좋은 요소가 있다면 반영하되, 부족한 부분을 완성된 형태로 보완하세요.\n"
        "- 메타 설명('~를 언급해야 합니다', '~에 대해 설명하십시오') 형태로 작성하지 마세요. 반드시 완성된 모범 답변 문장을 작성하세요.\n"
        "- 분량은 3~5문장 내외로 핵심만 담아 작성하세요.\n\n"
        "【소프트스킬 채점 루브릭】\n"
        f"{rubric_guide}\n\n"
        "위 루브릭을 기준으로 각 역량을 채점하세요. score는 루브릭 구간에 맞게 정수로, "
        "evidence는 이 점수를 준 근거를 1~2문장으로 간결하게, weakness는 부족한 점을 1~2문장으로 간결하게 작성하세요. "
        "evidence와 weakness 모두 '질문 N번에서' 또는 '질문 N과 M에서'처럼 특정 질문 번호를 언급하지 마세요. "
        "전반적인 답변 경향을 기준으로 작성하세요.\n\n"
        "위 면접 전체를 분석하여 다음 JSON 형식으로만 응답하세요. JSON 외 다른 텍스트는 절대 포함하지 마세요:\n\n"
        "{\n"
        '  "summary": "면접 전체에 대한 2~3문장 총평 (실제 답변 품질을 솔직하게 반영)",\n'
        '  "softskill_analysis": {\n'
        f"{skill_schema_entries}\n"
        "  },\n"
        '  "questions": [\n'
        f"{question_schema}\n"
        "  ]\n}\n\n"
        "반드시 한국어로만 작성하고, 유효한 JSON만 출력하세요."
    )

    raw = await call_llm([
        {"role": "system", "content": "You are a Korean interview evaluator. Respond ONLY with valid JSON in Korean. No markdown, no explanation, just JSON. Evaluate strictly based on the actual answer provided. Do NOT fabricate or assume positive qualities that are not present in the answer."},
        {"role": "user", "content": prompt},
    ], timeout=120)

    try:
        return _extract_json(raw)
    except Exception:
        # 파싱 실패 시 기본 구조 반환
        return {
            "summary": "면접이 완료되었습니다.",
            "softskill_analysis": {
                c: {"score": 70, "evidence": "", "weakness": ""}
                for c in skill_keys
            },
            "questions": [
                {
                    "question_id": q.question_id,
                    "answer_summary": "",
                    "intent": "질문 의도를 분석할 수 없습니다.",
                    "feedback": "피드백을 생성할 수 없습니다.",
                    "improved_answer": "개선된 답변을 생성할 수 없습니다."
                }
                for q in req.questions
            ]
        }


# 알파벳 → 한국어 발음 매핑
_LETTER_KO = {
    'a':'에이','b':'비','c':'씨','d':'디','e':'이','f':'에프','g':'지',
    'h':'에이치','i':'아이','j':'제이','k':'케이','l':'엘','m':'엠',
    'n':'엔','o':'오','p':'피','q':'큐','r':'알','s':'에스','t':'티',
    'u':'유','v':'브이','w':'더블유','x':'엑스','y':'와이','z':'지',
}

# 기술 용어 사전 (소문자 키)
_TECH_KO = {
    "HTTP":"에이치티티피", "HTTPS":"에이치티티피에스",
    "api":"에이피아이", "apis":"에이피아이",
    "rest":"레스트", "graphql":"그래프큐엘",
    "sql":"에스큐엘", "nosql":"노에스큐엘",
    "html":"에이치티엠엘", "css":"씨에스에스",
    "url":"유알엘", "urls":"유알엘",
    "ui":"유아이", "ux":"유엑스",
    "db":"디비", "os":"오에스",
    "aws":"에이더블유에스", "gcp":"지씨피", "azure":"애저",
    "jwt":"제이더블유티", "oauth":"오어스",
    "ci":"씨아이", "cd":"씨디",
    "oop":"오오피", "mvc":"엠브이씨", "mvp":"엠브이피",
    "jvm":"제이브이엠", "jdk":"제이디케이", "jre":"제이알이",
    "sdk":"에스디케이", "ide":"아이디이",
    "cpu":"씨피유", "gpu":"지피유", "ram":"램",
    "ssh":"에스에스에이치", "ssl":"에스에스엘", "tls":"티엘에스",
    "ssd":"에스에스디", "hdd":"에이치디디",
    "json":"제이슨", "xml":"엑스엠엘", "yaml":"야믈",
    "grpc":"지알피씨", "tcp":"티씨피", "udp":"유디피", "ip":"아이피",
    "git":"깃", "github":"깃허브", "gitlab":"깃랩",
    "docker":"도커", "kubernetes":"쿠버네티스", "k8s":"케이에잇에스",
    "linux":"리눅스", "ubuntu":"우분투",
    "spring":"스프링", "react":"리액트", "vue":"뷰", "angular":"앵귤러",
    "nodejs":"노드제이에스", "django":"장고", "flask":"플라스크",
    "mysql":"마이에스큐엘", "postgresql":"포스트그레에스큐엘",
    "redis":"레디스", "mongodb":"몽고디비", "chromadb":"크로마디비",
    "kafka":"카프카", "rabbitmq":"래빗엠큐",
    "nginx":"엔진엑스", "apache":"아파치",
    "msa":"엠에스에이", "devops":"데브옵스",
    "backend":"백엔드", "frontend":"프론트엔드",
    "server":"서버", "client":"클라이언트",
    "framework":"프레임워크", "library":"라이브러리",
    "cloud":"클라우드", "serverless":"서버리스",
    "microservice":"마이크로서비스", "monolithic":"모놀리식",
    "performance":"퍼포먼스", "scalability":"확장성",
    "cache":"캐시", "caching":"캐싱",
    "deploy":"배포", "deployment":"배포",
    "whisper":"위스퍼", "edge-tts":"엣지 티티에스",
    "mediapipe":"미디어파이프", "pymupdf":"파이엠유피디에프",
    "get":"겟", "post":"포스트", "put":"풋", "delete":"딜리트",
    "ollama":"올라마", "exaone":"엑사원",
    "python":"파이썬", "java":"자바", "javascript":"자바스크립트",
    "typescript":"타입스크립트", "csharp":"씨샵", "cpp":"씨플플",
    "inner":"이너", "outer":"아우터", "join":"조인", "left":"레프트", "right":"라이트", "full":"풀",   
}

def _word_to_ko(word: str) -> str:
    """단어를 한국어 발음으로 변환. 사전 → 알파벳 한 글자씩 fallback."""
    lower = word.lower()
    if lower in _TECH_KO:
        return _TECH_KO[lower]
    # 알파벳을 한 글자씩 읽기 (예: "http" → "에이치 티 티 피")
    return " ".join(_LETTER_KO.get(c, c) for c in lower if c.isalpha())

def preprocess_tts_korean(text: str) -> str:
    """한국어 TTS 전달 전 영어 단어를 한국어 발음으로 치환."""
    # 괄호 안 텍스트 제거 (소괄호/대괄호/중괄호)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    # URL 전체를 먼저 제거 (주소 자체를 읽으면 너무 길어짐)
    text = re.sub(r'https?://\S+', '링크', text)
    # 영문+숫자 혼합 단어 처리 (예: Node.js, k8s, CI/CD)
    text = re.sub(r'\bNode\.js\b', '노드 제이에스', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCI/CD\b', '씨아이 씨디', text, flags=re.IGNORECASE)
    # 영문 단어 시퀀스 치환 (한국어 글자가 섞이지 않은 순수 영어 단어)
    text = re.sub(r'[a-zA-Z][a-zA-Z0-9]*', lambda m: _word_to_ko(m.group()), text)
    return text


class TTSRequest(BaseModel):
    text: str
    voice: str = "ko-KR-SunHiNeural"  # edge-tts 보이스 이름

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """텍스트를 edge-tts로 변환해 MP3 오디오 스트림 반환."""
    tts_text = preprocess_tts_korean(req.text) if req.voice.startswith("ko-KR") else req.text
    communicate = edge_tts.Communicate(tts_text, req.voice, rate="+10%")
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


@app.post("/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    language: str = Form("ko"),
    desired_job: str = Form(""),
):
    """음성 파일을 faster-whisper로 텍스트 변환해 반환. 음성 분석 지표도 함께 반환."""
    suffix = Path(audio.filename).suffix if audio.filename else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        lang_code = language if language in ("ko", "en") else "ko"
        initial_prompt = _get_stt_prompt(desired_job, lang_code)
        loop = asyncio.get_event_loop()
        def run_transcribe():
            segs, _ = whisper_model.transcribe(
                tmp_path,
                language=lang_code,
                initial_prompt=initial_prompt,
                beam_size=5,
                vad_filter=True,
            )
            segs_list = list(segs)
            text = " ".join(seg.text for seg in segs_list).strip()

            # 음성 분석 지표 계산
            speech_duration = round(sum(seg.end - seg.start for seg in segs_list), 2)
            total_duration = round(
                (segs_list[-1].end - segs_list[0].start) if segs_list else 0.0, 2
            )
            # 한국어 음절 수(가-힣 문자 수) 기반 SPM 계산
            syllable_count = len(re.findall(r'[가-힣]', text))
            spm = round(syllable_count / speech_duration * 60) if speech_duration > 0 else 0

            return text, speech_duration, total_duration, spm

        text, speech_duration, total_duration, spm = \
            await loop.run_in_executor(None, run_transcribe)
        if lang_code == "ko":
            text = correct_stt_text(text)
    finally:
        os.unlink(tmp_path)
    return {
        "text": text,
        "voice_data": {
            "speechDuration": speech_duration,
            "totalDuration": total_duration,
            "spm": spm,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
