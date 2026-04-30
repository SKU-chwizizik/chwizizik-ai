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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import edge_tts
import io
import re

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
    "기술깊이":    {"score": 72, "evidence": "HTTP 메서드와 DB 차이 등 기본 개념을 대체로 정확히 설명했습니다.", "weakness": "일부 답변에서 내부 동작 원리보다 표면적 차이에 그쳐 깊이가 아쉬웠습니다."},
    "문제해결력":  {"score": 68, "evidence": "프로젝트 경험을 연결해 문제 상황을 설명하려는 시도가 있었습니다.", "weakness": "해결 과정의 논리적 흐름과 트레이드오프 고려가 부족했습니다."},
    "커뮤니케이션":{"score": 75, "evidence": "답변 구조가 비교적 명확했고 전달 흐름이 자연스러웠습니다.", "weakness": "기술 용어를 쉽게 풀어 설명하는 부분이 부족했습니다."},
    "논리적사고":  {"score": 70, "evidence": "질문 의도를 파악하고 관련 개념을 연결하는 능력을 보였습니다.", "weakness": "결론 도출 과정에서 근거가 약한 부분이 있었습니다."},
    "성장가능성":  {"score": 80, "evidence": "기술 선택 이유를 스스로 설명할 수 있는 자기주도적 학습 태도가 보였습니다.", "weakness": "최신 기술 트렌드에 대한 관심을 더 구체적으로 표현하면 좋겠습니다."},
}

SOFTSKILL_BASIC = {
    "커뮤니케이션": {"score": 75, "evidence": "자신의 경험을 비교적 논리적으로 전달했습니다.", "weakness": "상대방 관점을 고려한 표현이 부족했습니다."},
    "조직적합성":   {"score": 72, "evidence": "팀 내 갈등 상황에서 조율 역할을 맡으려 했다는 점이 긍정적입니다.", "weakness": "조직 문화에 맞는 구체적인 행동 사례가 더 필요합니다."},
    "문제해결력":   {"score": 68, "evidence": "어려운 상황에서 대안을 찾으려는 태도가 드러났습니다.", "weakness": "문제 원인 분석 과정이 다소 단순했습니다."},
    "리더십":       {"score": 65, "evidence": "팀 내 의견 조율 경험을 언급했습니다.", "weakness": "리더로서 주도적으로 방향을 제시한 구체적 사례가 부족했습니다."},
    "성장가능성":   {"score": 80, "evidence": "5년 후 목표와 현재 노력이 연결되어 있어 성장 의지가 느껴졌습니다.", "weakness": "목표 달성을 위한 구체적 실행 계획을 제시하면 더욱 좋겠습니다."},
}

# 질문 텍스트 → 피드백 매핑 (질문 텍스트 앞부분으로 매칭)
FEEDBACK_BY_QUESTION: dict[str, dict] = {
    # JOB
    "HTTP 메서드": {
        "intent": "HTTP 프로토콜의 기본 개념과 RESTful API 설계에 대한 이해도를 파악하기 위한 질문입니다.",
        "answer_summary": "GET은 조회, POST는 생성, PUT은 수정, DELETE는 삭제라고 설명하며 각 메서드의 기본 역할 차이를 언급했습니다.",
        "feedback": "GET과 POST의 차이는 잘 설명하셨지만, PUT과 PATCH의 구분, 멱등성(idempotency) 개념까지 연결하셨다면 더욱 완성도 있는 답변이 되었을 것입니다.",
        "improved_answer": "GET은 리소스 조회, POST는 생성, PUT은 전체 수정, DELETE는 삭제에 사용합니다. 특히 GET·PUT·DELETE는 멱등성을 가져 동일 요청을 반복해도 결과가 같지만, POST는 매번 새 리소스를 생성합니다. 실제 프로젝트에서는 RESTful 설계 원칙에 따라 각 메서드의 의미를 지켜 API를 설계했습니다.",
    },
    "프로세스와 스레드": {
        "intent": "운영체제의 프로세스·스레드 개념과 동시성 처리에 대한 기초 지식을 확인하기 위한 질문입니다.",
        "answer_summary": "프로세스는 독립적인 메모리를 가지고 스레드는 메모리를 공유한다고 설명했습니다.",
        "feedback": "프로세스와 스레드의 메모리 공유 여부는 언급하셨지만, 컨텍스트 스위칭 비용이나 멀티스레드 환경의 동기화 문제(Race Condition, Deadlock)까지 이야기하셨다면 더 깊이 있는 답변이 됐을 것입니다.",
        "improved_answer": "프로세스는 독립된 메모리 공간을 가진 실행 단위이고, 스레드는 프로세스 내 메모리를 공유하는 실행 흐름입니다. 스레드는 생성 비용이 낮고 통신이 빠르지만, 공유 자원 접근 시 Race Condition이 발생할 수 있어 동기화가 필요합니다. 실무에서는 Java의 synchronized나 concurrent 패키지를 활용해 스레드 안전성을 확보했습니다.",
    },
    "RDB와 NoSQL": {
        "intent": "데이터베이스 선택 기준과 트레이드오프에 대한 실무적 판단 능력을 평가하기 위한 질문입니다.",
        "answer_summary": "RDB는 정형 데이터와 트랜잭션에 적합하고 NoSQL은 비정형 대용량 데이터에 유리하다고 설명했습니다.",
        "feedback": "두 DB의 기본적인 차이는 설명하셨으나, 어떤 프로젝트 상황에서 어떤 DB를 선택하고 왜 그 결정을 내렸는지 실제 경험과 연결하셨다면 더욱 설득력 있는 답변이 되었을 것입니다.",
        "improved_answer": "RDB는 스키마가 명확하고 트랜잭션이 중요한 금융·주문 시스템에 적합하고, NoSQL은 스키마가 유동적이거나 대용량 비정형 데이터를 빠르게 처리해야 할 때 유리합니다. 제 프로젝트에서는 사용자 정보와 주문 데이터는 MySQL로 관리하고, 세션·캐시 데이터는 Redis를 사용해 응답 속도를 개선했습니다.",
    },
    "가장 자신 있는 프로젝트": {
        "intent": "지원자의 실무 경험과 팀 내 기여도, 문제 해결 능력을 구체적으로 파악하기 위한 질문입니다.",
        "answer_summary": "AI 면접 서비스 프로젝트에서 백엔드를 담당하여 REST API 설계와 비동기 처리를 구현했다고 답변했습니다.",
        "feedback": "프로젝트 개요는 잘 설명하셨지만, 본인이 주도적으로 해결한 기술적 난관이나 팀에 기여한 구체적인 성과 수치를 함께 제시하셨다면 더욱 인상적인 답변이 됐을 것입니다.",
        "improved_answer": "제가 가장 자신 있는 프로젝트는 팀 프로젝트로 진행한 AI 면접 서비스입니다. 저는 백엔드 개발을 담당해 Spring Boot 기반의 REST API를 설계하고, 비동기 처리를 도입해 응답 시간을 40% 단축했습니다. 특히 동시 요청 처리 시 발생하는 DB 병목 문제를 인덱스 최적화로 해결한 경험이 기억에 남습니다.",
    },
    "기술 스택을 선택한 이유": {
        "intent": "기술 선택에 대한 주체적인 사고와 근거 있는 판단 능력을 평가하기 위한 질문입니다.",
        "answer_summary": "Spring Boot, MySQL, React를 선택했으며 팀원 친숙도와 생태계 안정성을 이유로 들었습니다.",
        "feedback": "기술 스택을 나열하셨지만, 각 기술을 선택한 구체적인 이유와 대안 기술과의 비교 과정을 설명하셨다면 기술 판단력을 더 잘 보여드릴 수 있었을 것입니다.",
        "improved_answer": "백엔드로 Spring Boot를 선택한 이유는 팀원 모두 Java에 익숙하고, 대규모 트래픽에 검증된 생태계가 있었기 때문입니다. 데이터베이스는 정형 데이터 구조와 트랜잭션 일관성이 필요해 MySQL을 선택했고, 프론트엔드는 컴포넌트 재사용성과 개발 속도를 위해 React를 선택했습니다.",
    },
    # BASIC
    "5년 후 목표": {
        "intent": "지원자의 커리어 방향성과 직무에 대한 진지한 고민 여부를 파악하기 위한 질문입니다.",
        "answer_summary": "5년 후 시니어 개발자를 목표로 하며 현재 백엔드와 시스템 설계를 공부하고 있다고 답변했습니다.",
        "feedback": "목표를 제시하셨지만, 현재의 노력과 5년 후 목표 사이의 연결고리를 더 구체적으로 설명하셨다면 실현 가능성과 진정성이 더 잘 전달됐을 것입니다.",
        "improved_answer": "5년 후에는 풀스택 개발 역량을 갖춘 시니어 개발자로 성장하고 싶습니다. 현재는 백엔드 개발에 집중하면서 시스템 설계와 성능 최적화를 공부하고 있습니다. 장기적으로는 팀을 기술적으로 이끌 수 있는 리드 개발자가 되는 것이 목표입니다.",
    },
    "본인의 강점이 이 직무와": {
        "intent": "지원자가 자신의 역량을 직무 요건과 얼마나 연결해 이해하고 있는지 평가하기 위한 질문입니다.",
        "answer_summary": "꼼꼼한 문서화와 빠른 문제 파악 능력이 강점이며, 팀 프로젝트에서 이를 발휘했다고 답변했습니다.",
        "feedback": "강점을 언급하셨지만, 그 강점이 실제 업무 상황에서 어떻게 발휘됐는지 구체적인 사례로 뒷받침하셨다면 더욱 설득력 있었을 것입니다.",
        "improved_answer": "저의 강점은 꼼꼼한 문서화와 빠른 문제 파악 능력입니다. 팀 프로젝트에서 API 명세를 체계적으로 관리해 팀원 간 커뮤니케이션 오류를 줄였고, 배포 직전 발견된 버그를 빠르게 원인 분석해 일정을 맞출 수 있었습니다. 이런 강점이 개발자 직무에서 팀 생산성 향상에 기여할 수 있다고 생각합니다.",
    },
    "가장 크게 성장했다고 느낀": {
        "intent": "지원자의 자기 성찰 능력과 어려움을 통해 배운 점을 파악하기 위한 질문입니다.",
        "answer_summary": "처음 API 서버를 혼자 설계·배포하며 설계의 중요성을 깨달았고, 이후 ERD와 명세를 먼저 작성하는 습관을 들였다고 답변했습니다.",
        "feedback": "성장 경험을 언급하셨지만, 성장 전후의 구체적인 변화나 수치, 그 경험이 이후 업무 방식에 미친 영향을 함께 설명하셨다면 더욱 인상적인 답변이 됐을 것입니다.",
        "improved_answer": "처음으로 혼자 API 서버를 설계하고 배포했을 때 가장 크게 성장했습니다. 초기에는 설계 없이 코딩부터 시작해 중간에 구조를 전면 수정해야 하는 상황이 발생했습니다. 이 경험으로 설계 단계의 중요성을 깨달았고, 이후에는 ERD와 API 명세를 먼저 작성하는 습관을 들여 개발 효율이 크게 향상됐습니다.",
    },
    "의견 충돌이 생겼을 때": {
        "intent": "갈등 상황에서의 소통 방식과 팀워크 능력을 평가하기 위한 질문입니다.",
        "answer_summary": "장단점 비교 문서를 작성해 팀 전체가 검토하는 자리를 마련하여 합의를 이끌었다고 답변했습니다.",
        "feedback": "갈등 상황을 해결했다고 하셨지만, 구체적으로 어떤 방식으로 의견 차이를 좁혔는지, 그 결과 팀 분위기나 프로젝트에 어떤 긍정적 변화가 있었는지 설명하셨다면 더 풍부한 답변이 됐을 것입니다.",
        "improved_answer": "팀 프로젝트에서 기술 스택 선택을 두고 의견이 나뉜 적이 있었습니다. 저는 각자의 주장을 정리해 장단점 비교 문서를 작성하고 팀 전체가 함께 검토하는 자리를 마련했습니다. 감정이 아닌 데이터 기반으로 논의한 결과 합의에 이를 수 있었고, 이후 의사결정 시 이 방식을 팀의 관례로 삼게 됐습니다.",
    },
    "업무 스타일이 다른 팀원": {
        "intent": "다양한 성향의 동료와 협업하는 유연성과 적응력을 확인하기 위한 질문입니다.",
        "answer_summary": "즉흥적인 팀원과 협업 시 계획은 본인이, 실행은 팀원이 담당하는 방식으로 역할을 분담해 조율했다고 답변했습니다.",
        "feedback": "상황을 설명하셨지만, 상대방의 스타일을 어떻게 파악하고 본인이 어떻게 맞춰갔는지 구체적인 행동을 더 상세히 설명하셨다면 협업 역량이 더 잘 드러났을 것입니다.",
        "improved_answer": "계획 없이 즉흥적으로 진행하는 팀원과 함께 일한 경험이 있습니다. 처음에는 마찰이 있었지만, 팀원의 강점이 빠른 실행력에 있다는 것을 파악하고 제가 계획과 문서화를 맡고 팀원이 실행을 담당하는 방식으로 역할을 조정했습니다. 결과적으로 서로의 약점을 보완하며 프로젝트를 성공적으로 마무리했습니다.",
    },
}


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

# ── TTS 한국어 전처리 ─────────────────────────────────────────────────────────

_LETTER_KO = {
    'a':'에이','b':'비','c':'씨','d':'디','e':'이','f':'에프','g':'지',
    'h':'에이치','i':'아이','j':'제이','k':'케이','l':'엘','m':'엠',
    'n':'엔','o':'오','p':'피','q':'큐','r':'알','s':'에스','t':'티',
    'u':'유','v':'브이','w':'더블유','x':'엑스','y':'와이','z':'지',
}

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
    lower = word.lower()
    if lower in _TECH_KO:
        return _TECH_KO[lower]
    return " ".join(_LETTER_KO.get(c, c) for c in lower if c.isalpha())

def preprocess_tts_korean(text: str) -> str:
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'https?://\S+', '링크', text)
    text = re.sub(r'\bNode\.js\b', '노드 제이에스', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCI/CD\b', '씨아이 씨디', text, flags=re.IGNORECASE)
    text = re.sub(r'[a-zA-Z][a-zA-Z0-9]*', lambda m: _word_to_ko(m.group()), text)
    return text


class TTSRequest(BaseModel):
    text: str
    voice: str = "ko-KR-SunHiNeural"


@app.get("/health")
async def health_check():
    return {"status": "ok", "mode": "mock"}


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """edge-tts로 실제 음성 합성 — Ollama 없이도 동작."""
    tts_text = preprocess_tts_korean(req.text) if req.voice.startswith("ko-KR") else req.text
    communicate = edge_tts.Communicate(tts_text, req.voice, rate="+50%")
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


@app.post("/stt")
async def stt(
    audio: UploadFile = File(...),  # noqa: ARG001
    language: str = Form("ko"),    # noqa: ARG001
):
    """STT 스텁 — 하드코딩된 텍스트 즉시 반환 (Whisper 없이 동작)."""
    return {
        "text": "저는 해당 개념을 실제 프로젝트에서 활용한 경험이 있습니다.",
        "voice_data": None,
    }


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
    """면접 종합 피드백 스텁 — 질문 텍스트 기반 하드코딩 피드백 반환."""
    softskill = SOFTSKILL_JOB if req.interview_type == "job" else SOFTSKILL_BASIC

    def _lookup(question_text: str) -> dict:
        for keyword, data in FEEDBACK_BY_QUESTION.items():
            if keyword in question_text:
                return data
        return {
            "intent": "지원자의 역량과 경험을 파악하기 위한 질문입니다.",
            "feedback": "핵심을 잘 파악하고 답변하셨습니다. 구체적인 사례를 더 들어 주셨다면 더욱 설득력 있었을 것입니다.",
            "improved_answer": "저는 해당 경험을 통해 문제를 분석하고 해결하는 능력을 키웠습니다. 구체적으로는 상황을 파악한 뒤 대안을 모색했고, 그 결과 팀과 함께 목표를 달성할 수 있었습니다.",
        }

    feedback_questions = []
    for q in req.questions:
        matched = _lookup(q.question_text)
        feedback_questions.append({
            "question_id": q.question_id,
            "intent": matched["intent"],
            "answer_summary": matched.get("answer_summary", ""),
            "feedback": matched["feedback"],
            "improved_answer": matched["improved_answer"],
        })

    return {
        "summary": "전반적으로 기본기가 탄탄하고 자신의 경험을 잘 전달하셨습니다. 다음 면접에서는 보다 구체적인 사례와 수치를 활용하면 더욱 인상적인 답변이 될 것입니다.",
        "softskill_analysis": softskill,
        "questions": feedback_questions,
    }
