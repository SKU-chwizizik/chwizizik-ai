"""
Mock AI Service — Ollama 없이 동작하는 개발용 스텁 서버

사용법:
    uvicorn mock_ai_service:app --reload --port 8000

실제 main.py 와 동일한 포트(8000)·동일한 엔드포인트를 제공하지만
LLM 호출 없이 하드코딩된 응답을 즉시 반환합니다.
ChromaDB·pymupdf4llm·Ollama 가 없어도 실행됩니다.

프론트엔드·백엔드 개발 및 UI 테스트 전용입니다.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Mock AI Service (no LLM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 스키마 ───────────────────────────────────────────────────────────────────

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
    interview_type: str


class GenerateGreetingRequest(BaseModel):
    interview_type: str


class ShouldFollowupRequest(BaseModel):
    interview_type: str
    question_text: str
    user_answer: str
    current_followup_count: int


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


# ── 하드코딩 데이터 ──────────────────────────────────────────────────────────

POOL_JOB = [
    {"category": "A", "text": "HTTP 메서드(GET, POST, PUT, DELETE)의 차이점을 설명해 주세요."},
    {"category": "A", "text": "프로세스와 스레드의 차이점을 설명해 주세요."},
    {"category": "A", "text": "RDB와 NoSQL의 차이점과 각각 어떤 상황에 적합한지 말씀해 주세요."},
    {"category": "B", "text": "가장 자신 있는 프로젝트에서 본인의 역할과 기여를 말씀해 주세요."},
    {"category": "B", "text": "프로젝트에서 사용한 기술 스택을 선택한 이유를 말씀해 주세요."},
]

POOL_BASIC = [
    {"category": "A", "text": "개발자로서 5년 후 목표가 있다면 말씀해 주세요."},
    {"category": "A", "text": "본인의 강점이 이 직무와 어떻게 연결된다고 생각하시나요?"},
    {"category": "A", "text": "개발하면서 가장 크게 성장했다고 느낀 경험을 말씀해 주세요."},
    {"category": "B", "text": "팀 프로젝트에서 의견 충돌이 생겼을 때 어떻게 해결하셨는지 말씀해 주세요."},
    {"category": "B", "text": "업무 스타일이 다른 팀원과 함께 일해야 했던 경험과 그 대처 방법을 말씀해 주세요."},
]

FOLLOWUP_JOB = "방금 말씀하신 내용에서 좀 더 구체적인 사례를 들어 주실 수 있으신가요?"
FOLLOWUP_BASIC = "그 경험을 통해 개인적으로 어떤 점을 배우셨는지 말씀해 주실 수 있으신가요?"

SOFTSKILL_JOB = {
    "기술깊이": 72,
    "문제해결력": 68,
    "커뮤니케이션": 75,
    "논리적사고": 70,
    "성장가능성": 80,
}

SOFTSKILL_BASIC = {
    "커뮤니케이션": 75,
    "조직적합성": 72,
    "문제해결력": 68,
    "리더십": 65,
    "성장가능성": 80,
}


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "mode": "mock"}


@app.post("/embed-text")
async def embed_text(req: EmbedTextRequest):
    """이력서 임베딩 스텁 — ChromaDB 저장 없이 성공 응답만 반환."""
    return {"user_id": req.user_id, "chunks_stored": 5}


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    """PDF 파싱 스텁 — 더미 Markdown 반환."""
    return {
        "filename": file.filename,
        "markdown": "## 이력서 (Mock)\n\n- 이름: 홍길동\n- 기술 스택: Java, Spring Boot, React\n- 프로젝트: 모의 프로젝트 A\n",
    }


@app.post("/embed-resume")
async def embed_resume(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    """이력서 PDF 임베딩 스텁 — 저장 없이 성공 응답만 반환."""
    return {
        "filename": file.filename,
        "user_id": user_id,
        "chunks_stored": 5,
        "markdown": "## 이력서 (Mock)\n\n- 이름: 홍길동\n- 기술 스택: Java, Spring Boot, React\n",
    }


@app.post("/search")
async def search(req: SearchRequest):
    """벡터 검색 스텁 — 빈 결과 반환."""
    return {"query": req.query, "results": []}


@app.post("/generate-pool")
async def generate_pool(req: GeneratePoolRequest):
    """질문 풀 5개 생성 스텁 — 하드코딩된 질문 즉시 반환."""
    questions = POOL_JOB if req.interview_type == "job" else POOL_BASIC
    return {"questions": questions}


@app.post("/generate-greeting")
async def generate_greeting(req: GenerateGreetingRequest):
    """면접 시작 인사말 스텁."""
    if req.interview_type == "job":
        greeting = "안녕하세요, 개발팀 김 팀장입니다. 오늘 기술 면접에 참여해 주셔서 감사합니다. 그럼 시작하겠습니다."
    else:
        greeting = "안녕하세요, 박부장입니다. 오늘 면접에 와 주셔서 반갑습니다. 편안하게 이야기 나눠 봅시다."
    return {"greeting": greeting}


@app.post("/should-followup")
async def should_followup(req: ShouldFollowupRequest):
    """꼬리질문 필요 여부 스텁 — 항상 false 반환 (면접 흐름 단순화)."""
    return {"should_followup": False}


@app.post("/generate-followup")
async def generate_followup(req: GenerateFollowupRequest):
    """꼬리질문 생성 스텁."""
    text = FOLLOWUP_JOB if req.interview_type == "job" else FOLLOWUP_BASIC
    return {"question": text}


@app.post("/generate-closing")
async def generate_closing(req: GenerateClosingRequest):
    """면접 마무리 멘트 스텁."""
    if req.interview_type == "job":
        closing = "수고하셨습니다. 오늘 면접 결과는 추후 안내해 드리겠습니다."
    else:
        closing = "오늘 면접에 임해 주셔서 감사합니다. 좋은 결과가 있기를 바랍니다."
    return {"closing": closing}


@app.post("/evaluate-answer")
async def evaluate_answer(req: EvaluateAnswerRequest):
    """답변 품질 평가 스텁 — 항상 GOOD 반환."""
    return {"quality": "GOOD"}


@app.post("/generate-feedback")
async def generate_feedback(req: GenerateFeedbackRequest):
    """면접 종합 피드백 스텁 — 하드코딩된 피드백 반환."""
    softskill = SOFTSKILL_JOB if req.interview_type == "job" else SOFTSKILL_BASIC

    feedback_questions = []
    for q in req.questions:
        feedback_questions.append({
            "question_id": q.question_id,
            "intent": "지원자의 기술적 이해도와 실무 경험을 파악하기 위한 질문입니다.",
            "feedback": "전반적으로 핵심을 잘 파악하고 답변하셨습니다. 구체적인 사례를 더 들어 주셨다면 더욱 설득력 있었을 것입니다.",
            "improved_answer": "네, 저는 해당 개념을 실제 프로젝트에서 활용한 경험이 있습니다. 구체적으로는 [상황]에서 [조치]를 취했고, 그 결과 [성과]를 얻을 수 있었습니다.",
        })

    return {
        "summary": "전반적으로 기본기가 탄탄하고 자신의 경험을 잘 전달하셨습니다. 다음 면접에서는 보다 구체적인 사례와 수치를 활용하면 더욱 인상적인 답변이 될 것입니다.",
        "softskill_analysis": softskill,
        "questions": feedback_questions,
    }
