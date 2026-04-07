from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import edge_tts
import pymupdf4llm
import chromadb
import httpx
import io
import os
import shutil
import uuid

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


# ── 스키마 ───────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    top_k: int = 5


class EmbedTextRequest(BaseModel):
    user_id: str
    text: str
    filename: str = "resume"


class RagChatRequest(BaseModel):
    user_id: str
    interview_type: str          # "basic" | "job"
    last_question: str
    user_answer: str
    interview_id: int | None = None


class GeneratePoolRequest(BaseModel):
    user_id: str
    interview_type: str          # "basic" | "job"


class GenerateGreetingRequest(BaseModel):
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


class GenerateClosingRequest(BaseModel):
    interview_type: str


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

    try:
        embeddings = [await get_embedding(chunk) for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"user_id": req.user_id, "filename": req.filename, "chunk_index": i}
            for i, _ in enumerate(chunks)
        ]
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 실패: {e}")

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
    try:
        embeddings = []
        for chunk in chunks:
            emb = await get_embedding(chunk)
            embeddings.append(emb)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"user_id": user_id, "filename": file.filename, "chunk_index": i}
            for i, _ in enumerate(chunks)
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다. ollama serve 를 먼저 실행하세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩/저장 실패: {e}")

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


@app.post("/rag-chat")
async def rag_chat(req: RagChatRequest):
    """
    RAG 기반 면접 질문 생성:
    1. 유저 답변 기반 ChromaDB 이력서 청크 검색
    2. 시스템 프롬프트 + 이력서 컨텍스트 조합
    3. LLM(Ollama)으로 다음 질문 생성
    """
    system_prompt = JOB_SYSTEM if req.interview_type == "job" else BASIC_SYSTEM

    # 1. 이력서 컨텍스트 검색 (답변 + 이전 질문으로 검색)
    search_query = f"{req.last_question} {req.user_answer}"
    resume_context = await retrieve_resume_context(req.user_id, search_query)

    # 2. 이력서 정보가 있으면 시스템 프롬프트에 주입
    if resume_context:
        system_prompt += (
            "\n\n[지원자 이력서 발췌]\n"
            + resume_context
            + "\n위 이력서 정보를 참고하여 지원자의 실제 경험에 맞는 구체적인 질문을 하세요."
        )

    # 3. LLM 호출
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"면접관의 이전 질문: {req.last_question}\n지원자 답변: {req.user_answer}",
        },
    ]

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": LLM_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            next_question = data["message"]["content"]
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다. ollama serve 를 먼저 실행하세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solar 호출 실패: {e}")

    # 4. 면접 종료 감지
    is_finished = "[면접 종료]" in next_question
    if is_finished:
        next_question = next_question.replace("[면접 종료]", "").strip()

    return {
        "question": next_question,
        "isFinished": is_finished,
        "context_used": bool(resume_context),
    }


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
    import re

    resume_context = await retrieve_resume_context(req.user_id, "프로젝트 기술스택 경험")
    if not resume_context:
        raise HTTPException(status_code=400, detail="이력서 정보를 찾을 수 없습니다. 마이페이지에서 이력서를 먼저 업로드해 주세요.")

    system_msg = "You are a Korean interviewer. You MUST respond only in Korean (한국어). Never use English, Japanese, Chinese, or any other language. Use correct Korean spelling and spacing."

    if req.interview_type == "job":
        persona = "17년 차 수석 개발자 면접관 '개발팀 김 팀장'"
        category_specs = [
            ("A", "CS 기초 개념 질문 — 다음 중 하나 선택: HTTP 메서드/상태코드, 프로세스 vs 스레드, RDB vs NoSQL, DB 인덱스 역할, TCP vs UDP. 개념을 말로 설명하는 수준이며 코드 구현 요구 절대 금지. 신입 지원자가 답할 수 있는 난이도."),
            ("A", "CS 기초 개념 질문 — 앞 질문과 다른 분야 선택: GC 가비지 컬렉션, HTTP vs HTTPS, RESTful API 개념, 캐시의 역할, OSI 7계층 중 하나. 개념 설명 수준, 코드 구현 요구 절대 금지."),
            ("A", "CS 기초 개념 질문 — 앞 두 질문과 다른 분야 선택: 동기 vs 비동기, 트랜잭션 ACID, 세션 vs JWT, 로드밸런싱 개념, CI/CD 개념 중 하나. 개념 설명 수준, 코드 구현 요구 절대 금지."),
            ("B", "프로젝트/기술 역량 질문 — 이력서의 특정 프로젝트 기능을 어떤 방식으로 구현했는지·왜 그 방식을 선택했는지, 또는 현재 관심 있는 기술이나 최근 공부한 내용을 묻는 질문"),
            ("B", "프로젝트 심화 질문 — 이력서 프로젝트에서 가장 도전적이었던 부분과 해결 방법, 또는 팀 프로젝트 실패 경험과 교훈, 또는 버그 발견·테스트 프로세스를 묻는 질문. 앞 질문과 다른 주제."),
        ]
    else:
        persona = "23년 차 임원 면접관 '박부장'"
        category_specs = [
            ("A", "인성 질문 — 다음 중 하나 선택: 개발자로서 5년/10년 후 목표, 개발이 적성에 맞는 이유, 개발자에게 가장 중요한 역량, 주니어와 시니어 개발자의 차이, 압박감이 클 때 일하는 방식. 지원자의 가치관과 성장 의지를 파악하는 질문."),
            ("A", "인성 질문 — 앞 질문과 다른 주제 선택: 자신의 역량이 직무와 맞는 이유, 본인의 강점과 약점, 실패 경험과 교훈, 개발 외 자기계발 방법 중 하나."),
            ("A", "이력서/경험 기반 인성 질문 — 지원자의 실제 프로젝트·스터디·활동에서 배운 점이나 성취 경험을 묻는 질문. 앞 두 질문과 다른 주제."),
            ("B", "팀워크/소프트스킬 질문 — 다음 중 하나 선택: 팀 내 갈등 해결 경험, 의견 충돌 시 팀원 설득 방법, 팀에서 본인의 역할, 효과적인 팀 커뮤니케이션 전략 중 하나."),
            ("B", "팀워크/소프트스킬 질문 — 앞 질문과 다른 주제 선택: 업무 스타일이 다른 팀원과 협업 방법, 다른 개발자와 협력해 문제를 해결한 경험, 스트레스·압박 상황 대처 방식 중 하나."),
        ]

    resume_section = ""
    if resume_context:
        resume_section = f"\n[지원자 이력서 발췌 - 참고용]\n{resume_context}\n"

    rules = (
        "규칙: "
        "1) 질문은 1문장으로 짧고 간결하게. "
        "2) ArrayList, Spring Boot 등 고유 기술 용어는 영어 가능. "
        "3) 나머지는 모두 한국어. "
        "4) 자연스러운 면접 질문 형식으로 끝낼 것 (예: ~말씀해 주세요., ~설명해 주세요., ~있으신가요?). "
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


@app.post("/generate-greeting")
async def generate_greeting(req: GenerateGreetingRequest):
    """면접 시작 인사말 생성 (2~3문장)."""
    if req.interview_type == "job":
        prompt = (
            "반드시 한국어로만 작성하고, 영어 번역이나 영어 설명을 절대 포함하지 마세요. "
            "당신은 '개발팀 김 팀장'입니다. "
            "오늘 기술 면접을 시작하는 짧고 격식 있는 인사말을 2~3문장으로 작성하세요. "
            "면접관 소개와 오늘 면접의 방향(기술 역량 평가)을 간략히 언급하세요. "
            "인사말만 출력하고 다른 내용은 포함하지 마세요."
        )
    else:
        prompt = (
            "반드시 한국어로만 작성하고, 영어 번역이나 영어 설명을 절대 포함하지 마세요. "
            "당신은 '박부장'입니다. "
            "오늘 임원 면접을 시작하는 따뜻하고 여유로운 인사말을 2~3문장으로 작성하세요. "
            "면접관 소개와 편안한 분위기를 조성하는 말을 포함하세요. "
            "인사말만 출력하고 다른 내용은 포함하지 마세요."
        )

    greeting = await call_llm([
        {"role": "system", "content": "당신은 한국인 면접관입니다. 반드시 한국어로만 답변하세요. 다른 언어는 사용하지 않습니다."},
        {"role": "user", "content": prompt},
    ])
    return {"greeting": greeting.strip()}


@app.post("/should-followup")
async def should_followup(req: ShouldFollowupRequest):
    """꼬리 질문 여부 판단. current_followup_count >= 2이면 즉시 false 반환."""
    import json

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
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return {"should_followup": bool(parsed.get("should_followup", False))}
    except Exception:
        return {"should_followup": False}


@app.post("/generate-followup")
async def generate_followup(req: GenerateFollowupRequest):
    """꼬리 질문 생성 (부모 질문 + 사용자 답변 컨텍스트)."""
    if req.interview_type == "job":
        persona = "17년 차 수석 개발자 면접관 '개발팀 김 팀장'"
    else:
        persona = "23년 차 임원 면접관 '박부장'"

    prompt = (
        f"당신은 {persona}입니다.\n"
        f"면접관 질문: {req.parent_question}\n"
        f"지원자 답변: {req.user_answer}\n\n"
        "위 답변에서 더 구체적으로 파고들 부분을 찾아 꼬리 질문 1개를 작성하세요. "
        "질문만 출력하고 다른 내용은 포함하지 마세요. 존댓말로 작성하세요. "
        "반드시 한국어로만 작성하고, 영어 번역이나 설명을 절대 포함하지 마세요."
    )

    followup = await call_llm([
        {"role": "system", "content": "You are a Korean interviewer. You MUST respond only in Korean (한국어). Never use English, Japanese, Chinese, or any other language. Use correct Korean spelling and spacing."},
        {"role": "user", "content": prompt},
    ])
    return {"followup_question": followup.strip()}


@app.post("/generate-closing")
async def generate_closing(req: GenerateClosingRequest):
    """면접 마무리 멘트 생성 (2~3문장)."""
    if req.interview_type == "job":
        prompt = (
            "반드시 한국어로만 작성하고, 영어 번역이나 영어 설명을 절대 포함하지 마세요. "
            "당신은 '개발팀 김 팀장'입니다. "
            "기술 면접이 끝났습니다. 짧고 격식 있는 마무리 멘트를 2~3문장으로 작성하세요. "
            "수고했다는 말과 결과 안내 관련 내용을 포함하세요. "
            "마무리 멘트만 출력하세요."
        )
    else:
        prompt = (
            "반드시 한국어로만 작성하고, 영어 번역이나 영어 설명을 절대 포함하지 마세요. "
            "당신은 '박부장'입니다. "
            "임원 면접이 끝났습니다. 따뜻하고 격려하는 마무리 멘트를 2~3문장으로 작성하세요. "
            "수고했다는 말과 앞으로의 기대를 담아 주세요. "
            "마무리 멘트만 출력하세요."
        )

    closing = await call_llm([
        {"role": "system", "content": "You are a Korean interviewer. You MUST respond only in Korean (한국어). Never use English, Japanese, Chinese, or any other language. Use correct Korean spelling and spacing."},
        {"role": "user", "content": prompt},
    ])
    return {"closing": closing.strip()}


@app.post("/evaluate-answer")
async def evaluate_answer(req: EvaluateAnswerRequest):
    """
    답변 품질 평가 → "GOOD" | "POOR" 반환.
    평가 기준: 답변의 구체성, 완결성, 질문 관련성.
    """
    import json

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
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        quality = "POOR" if parsed.get("should_retry", False) else "GOOD"
    except Exception:
        quality = "GOOD"

    return {"quality": quality}


@app.post("/generate-feedback")
async def generate_feedback(req: GenerateFeedbackRequest):
    """
    면접 종합 피드백 생성.
    - 면접 총평 (summary)
    - 소프트스킬 카테고리별 점수 (softskill_analysis)
    - 질문별: 질문 의도, 답변 피드백, 개선된 답변
    """
    import json

    interviewer = "개발팀 김 팀장 (17년 차 수석 개발자)" if req.interview_type == "job" else "박부장 (23년 차 임원)"
    skill_categories = '["기술깊이", "문제해결력", "커뮤니케이션", "논리적사고", "성장가능성"]' if req.interview_type == "job" \
        else '["커뮤니케이션", "조직적합성", "문제해결력", "리더십", "성장가능성"]'

    questions_text = "\n\n".join(
        f"[질문 {q.question_id}]\n질문: {q.question_text}\n답변: {q.answer_text}"
        for q in req.questions
    )

    skill_keys = json.loads(skill_categories)
    skill_schema = ", ".join(f'"{c}": 점수(0~100)' for c in skill_keys)
    question_schema_lines = []
    for q in req.questions:
        question_schema_lines.append(
            '    {"question_id": ' + str(q.question_id) +
            ', "intent": "이 질문의 의도", "feedback": "이 답변에 대한 구체적 피드백", "improved_answer": "더 나은 모범 답변"}'
        )
    question_schema = ",\n".join(question_schema_lines)

    prompt = (
        f"당신은 면접관 '{interviewer}'입니다.\n"
        f"아래는 면접에서 오간 질문과 답변입니다.\n\n"
        f"{questions_text}\n\n"
        "위 면접 전체를 분석하여 다음 JSON 형식으로만 응답하세요. JSON 외 다른 텍스트는 절대 포함하지 마세요:\n\n"
        "{\n"
        '  "summary": "면접 전체에 대한 2~3문장 총평",\n'
        f'  "softskill_analysis": {{{skill_schema}}},\n'
        '  "questions": [\n'
        f"{question_schema}\n"
        "  ]\n}\n\n"
        "반드시 한국어로만 작성하고, 유효한 JSON만 출력하세요."
    )

    raw = await call_llm([
        {"role": "system", "content": "You are a Korean interview evaluator. Respond ONLY with valid JSON in Korean. No markdown, no explanation, just JSON."},
        {"role": "user", "content": prompt},
    ], timeout=120)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return parsed
    except Exception:
        # 파싱 실패 시 기본 구조 반환
        return {
            "summary": "면접이 완료되었습니다.",
            "softskill_analysis": {c: 70 for c in json.loads(skill_categories)},
            "questions": [
                {
                    "question_id": q.question_id,
                    "intent": "질문 의도를 분석할 수 없습니다.",
                    "feedback": "피드백을 생성할 수 없습니다.",
                    "improved_answer": "개선된 답변을 생성할 수 없습니다."
                }
                for q in req.questions
            ]
        }


class TTSRequest(BaseModel):
    text: str
    voice: str = "ko-KR-SunHiNeural"  # edge-tts 보이스 이름

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """텍스트를 edge-tts로 변환해 MP3 오디오 스트림 반환."""
    communicate = edge_tts.Communicate(req.text, req.voice, rate="+10%")
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
