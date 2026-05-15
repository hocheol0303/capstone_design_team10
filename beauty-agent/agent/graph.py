import inspect

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.prompts import SYSTEM_PROMPT
from config import OPENAI_MODEL
# from tools.search_pubmed import search_pubmed
from tools.skin_analyze import skin_analyze


def build_agent():
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    # tools = [skin_analyze, search_pubmed]
    tools = [skin_analyze]

    # langgraph 0.2+ uses `prompt`; older versions used `state_modifier`.
    params = inspect.signature(create_react_agent).parameters
    if "prompt" in params:
        return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
    return create_react_agent(model=llm, tools=tools, state_modifier=SYSTEM_PROMPT)


def run_agent(image_path: str, text: str) -> dict:
    agent = build_agent()

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"[이미지 경로: {image_path}]\n{text}",
            }
        ]
    })

    return {
        "final_answer": result["messages"][-1].content,
        "messages": result["messages"],
    }


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    txt = sys.argv[2] if len(sys.argv) > 2 else "어떤 시술을 받아야 할까요?"
    out = run_agent(img, txt)
    print(out["final_answer"])
