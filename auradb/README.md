# VLM agent

이 저장소는 엑셀에 있는 뷰티 룰 데이터를 업로드하고, Neo4j와 연동하는 Python 스크립트를 포함하고 있습니다.

## 준비 사항

- Python 3.10 이상
- 접속 가능한 Neo4j 인스턴스
- OpenAI API 키

## 실행 전 설정

1. 가상환경을 만들고 활성화합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 필요한 패키지를 설치합니다.

```powershell
pip install -r requirements.txt
```

3. 환경변수 예시 파일을 복사한 뒤, 실제 값으로 수정합니다.

```powershell
Copy-Item .env.example .env
```

`.env` 파일에서 아래 항목들을 실제 값으로 채워주세요.

- `OPENAI_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `EXCEL_PATH`

## 실행 방법

엑셀 데이터를 업로드하려면 아래 명령어를 실행합니다.

```powershell
python .\aura_upload_beauty_rules.py
```

Neo4j 예제 연결 스크립트를 실행하려면 아래 명령어를 사용합니다.

```powershell
python .\Connect_DB.py
```

## 처음 받는 사람이 가장 쉽게 실행하는 순서

아래 순서대로 진행하면 됩니다.

```powershell
git clone <깃허브_저장소_URL>
cd <저장소_폴더명>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python .\aura_upload_beauty_rules.py
```

실행 전에 `.env` 파일 값과 `EXCEL_PATH` 경로가 본인 환경에 맞는지 꼭 확인해주세요.

## 안전하게 공유하기

- `.env.example`만 커밋하고 `.env`는 커밋하지 않습니다.
- 소스코드에 들어 있었던 API 키나 DB 비밀번호는 반드시 새 값으로 교체하는 것을 권장합니다.
- `data/` 폴더의 엑셀 파일이 꼭 필요하다면, 공유해도 되는 파일인지 확인한 뒤 함께 올려주세요.

## GitHub 업로드 방법

아직 이 폴더가 Git 저장소가 아니라면 아래 순서로 업로드할 수 있습니다.

```powershell
git init
git add .
git commit -m "Initial project setup"
git branch -M main
git remote add origin <깃허브_저장소_URL>
git push -u origin main
```
