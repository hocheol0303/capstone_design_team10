from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# .env 파일에서 환경 변수 로드
load_dotenv()

# init_chat_model 사용 (권장)
llms = [
    init_chat_model("openai:gpt-5-mini", temperature=0),
    init_chat_model("google_genai:gemini-3-flash-preview", temperature=0)
    ]

response = llms[1].invoke([HumanMessage(content="LangGraph의 핵심 개념을 설명해주세요.")])
print(response.content)