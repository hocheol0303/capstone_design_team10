"""beauty-agent 통합 엔트리.

실행 위치: 프로젝트 루트 (auradb/, beauty-agent/, pipeline/ 의 부모)

사용 예
-------
    # 1) 일회성 진단
    python main.py samples/028_data/0001/0001_01_F.jpg
    python main.py samples/028_data/0001/0001_01_F.jpg female

    # 2) 대화형
    python main.py --chat
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BEAUTY_AGENT = ROOT / "beauty-agent"
if str(BEAUTY_AGENT) not in sys.path:
    sys.path.insert(0, str(BEAUTY_AGENT))


def one_shot(image_path: str, gender: str | None) -> None:
    from agent.graph import run_agent_once

    parts = [f"이 사진의 피부 상태를 진단해줘. 이미지 경로: {image_path}"]
    if gender:
        parts.append(f"성별: {gender}")
    out = run_agent_once("\n".join(parts))
    print("\n=== Final Answer ===")
    print(out["final_answer"])


def chat() -> None:
    from agent.graph import ChatSession

    print("대화형 모드 시작. 'exit' 입력 시 종료.")
    print("예) 이 사진 진단해줘. 성별은 여자, 경로: samples/028_data/0001/0001_01_F.jpg")
    session = ChatSession()
    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        reply = session.send(text)
        print(f"\nAgent> {reply}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    if args[0] == "--chat":
        chat()
    else:
        image_path = args[0]
        gender = args[1] if len(args) > 1 else None
        one_shot(image_path, gender)


if __name__ == "__main__":
    main()
