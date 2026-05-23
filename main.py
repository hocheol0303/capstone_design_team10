"""beauty-agent 통합 엔트리 (채팅 전용).

실행 위치: 프로젝트 루트 (auradb/, beauty-agent/, pipeline/ 의 부모)

사용
----
    python main.py --chat
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BEAUTY_AGENT = ROOT / "beauty-agent"
if str(BEAUTY_AGENT) not in sys.path:
    sys.path.insert(0, str(BEAUTY_AGENT))


def chat() -> None:
    from agent.graph import ChatSession

    print("대화형 모드 시작. 'exit' 입력 시 종료.")
    print("skin_analyze → recommend_treatment_db 순차 호출 예:\n이 사진을 보고 어떤 시술을 받을지 추천해줘. 성별은 여자, 경로: samples/028_data/0001/0001_01_F.jpg")
    session = ChatSession()
    while True:
        try:
            print("\n\033[92mYou>\033[0m ")
            text = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break

        print("\n\033[94mAgent>\n\033[0m", end="", flush=True)
        for token in session.stream(text):
            print(token, end="", flush=True)
        print()


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0)
    chat()


if __name__ == "__main__":
    main()
