# Beauty Agent — 얼굴 진단 + 시술 추천 에이전트

얼굴 사진과 자연어 대화를 입력 받아 부위별 피부 상태를 진단하고, AuraDB(Neo4j)에 적재된 시술 규칙에서 결정론적으로 매칭된 시술을 추천하는 LangGraph 기반 ReAct 에이전트.

```
   사용자 ──▶ main.py (ChatSession)
                │
                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │   beauty-agent/agent/  ReAct StateGraph (책 정석 패턴)       │
   │       graph.py                                              │
   │                                                             │
   │   START → think → ┬─ act(ToolNode) → observe ─┬→ think      │
   │                   │                          │             │
   │                   └─ (tool_call 없음)        └─→ finish → END│
   │                                                             │
   │   think:   LLM이 thought + tool_call 결정                   │
   │   act:     ToolNode가 last AIMessage의 tool_calls 실행      │
   │   observe: 최신 ToolMessage를 observations에 적재           │
   │   finish:  final_answer 적재 + is_complete=True             │
   └─────┬───────────────┬───────────────────────┬──────────────┘
         │               │                       │
         ▼               ▼                       ▼
   ┌───────────────┐  ┌──────────────────┐  ┌────────────────────┐
   │  pipeline/    │  │  auradb/ (Neo4j) │  │  PubMed E-utilities │
   │  TFLite 추론  │  │  FeatureRule +   │  │  esearch+efetch +   │
   │  (얼굴 분석)  │  │  rule_matcher    │  │  OpenAI embedding   │
   │               │  │  결정론적 매칭   │  │  cosine 랭킹 (RAG)  │
   └───────────────┘  └──────────────────┘  └────────────────────┘
          ▲                   ▲
          │                   │
   classification/      mapping_table_tmp.xlsx
   (TFLite 학습)       (시술 규칙 마스터)
```

두 추천 소스(`auradb`, `PubMed`)는 **독립**적으로 호출된다. 향후 final report 단계에서
가중치 기반 debate(예: 6:4)로 통합 예정.

## 디렉터리 구조

| 경로 | 역할 |
|---|---|
| [main.py](main.py) | 채팅 CLI 진입점. `ChatSession`을 생성하고 token-stream UI 제공 |
| [beauty-agent/](beauty-agent/) | LangGraph 에이전트 본체 (그래프, prompts, tools) |
| [pipeline/](pipeline/) | 얼굴 이미지 → TFLite 추론 파이프라인 (age/pigment/wrinkle/homogenity/sagging) |
| [classification/](classification/) | TFLite 모델 학습·추론·변환 코드 (EfficientNet 등) |
| [auradb/](auradb/) | Neo4j AuraDB 적재 스크립트, 조회 헬퍼, 규칙 기반 매처 |
| [docker/](docker/) | CPU 전용 런타임 이미지 Dockerfile |
| [samples/](samples/) | 테스트용 얼굴 이미지 (028_data, 034_data, sample_crop) |
| `rag/`, `vlm/` | placeholder (미사용) |
| `no_track/` | 개인 작업 폴더 (.gitignore) |

---

## beauty-agent — 에이전트

**핵심 파일**

| 파일 | 역할 |
|---|---|
| [agent/graph.py](beauty-agent/agent/graph.py) | 책 정석 ReAct StateGraph. `think → act(ToolNode) → observe → conditional(think\|finish) → END` 사이클. `ChatSession`이 messages+updates 혼합 스트림을 받아 `🔧 [Act]` / `📋 [Observe]` 마커를 사용자에게 노출. iteration_count로 루프 가드 |
| [agent/state.py](beauty-agent/agent/state.py) | `BeautyAgentState(MessagesState)` — 책 표준 8필드(thoughts/actions/observations/iteration_count/max_iterations/current_goal/final_answer/is_complete) + 도메인 6필드(image_path/gender/skin_scores/top_concerns/db_recommendations/pubmed_recommendations) |
| [agent/prompts.py](beauty-agent/agent/prompts.py) | 시스템 프롬프트. 의도-매트릭스(진단/추천/동시), ReAct 규칙(한 응답에 도구 1개), 응답 작성 기준 |
| [tools/skin_analyze.py](beauty-agent/tools/skin_analyze.py) | 얼굴 이미지 → raw 0~100 점수 추출. 내부에서 [pipeline/SkinPipeline](pipeline/pipeline.py) 호출. 결과로 state.skin_scores + top_concerns 업데이트 |
| [tools/recommend_treatment_db.py](beauty-agent/tools/recommend_treatment_db.py) | state.skin_scores → AuraDB 조회 → 부위별 시술 매칭. region→feature_name 매핑, age 기반 age_group 자동 선택, [auradb/rule_matcher](auradb/rule_matcher.py) 결정론적 매칭 후 매칭 실패 시 벡터 fallback |
| [tools/search_pubmed.py](beauty-agent/tools/search_pubmed.py) | state.skin_scores → 점수 가장 낮은 top 3 부위에 대해 PubMed esearch + efetch → OpenAI 임베딩 cosine RAG 랭킹 → 부위별 top-1 논문(PMID/저자/연도/초록)을 state.pubmed_recommendations에 적재. auradb와 **독립** |
| [config.py](beauty-agent/config.py) | LLM 모델 ID, 경로 상수, .env 로드 |

**그래프 흐름 (책 정석 think/act/observe/finish)**

```
START ──▶ think ──┬─ tool_calls 있음 → act (ToolNode) ──▶ observe ──┬─ iter < max → think
                  │                                                  └─ iter ≥ max → finish
                  └─ tool_calls 없음 (최종 답변)                    ─▶ finish ──▶ END
```

- **think** (`agent/graph.py:think`): LLM이 SystemMessage + history 보고 thought 본문 + tool_call 결정.
  매 호출에 iteration_count +1. thoughts/actions 리스트에 흔적 append.
- **act**: `langgraph.prebuilt.ToolNode`가 last AIMessage의 tool_calls를 실행. 도구별 가드(image_path 필수,
  skin_scores 필수 등)는 각 도구 함수 본문에서 직접 수행.
- **observe**: 가장 최근 ToolMessage들을 observations 리스트에 append.
- **finish**: 마지막 AIMessage.content를 final_answer로 발췌 + is_complete=True.

라우팅:
- `route_after_think`: tool_call 있으면 `act`, 없으면 `finish`.
- `should_continue` (observe 다음): iteration_count ≥ max_iterations이면 `finish`, 아니면 `think`.

체이닝 자동: 추천/논문 요청 + skin_scores 없음 → think 1차에 skin_analyze 호출, observe 후 think 2차에서
recommend_treatment_db나 search_pubmed 호출, think 3차에서 최종 답변 작성 (한 turn에 자동 연쇄).

**State**

**책 정석 ReAct 필드**

| 필드 | 채워지는 시점 | 비고 |
|---|---|---|
| `messages` | 매 단계 (add_messages reducer, MessagesState 상속) | HumanMessage, AIMessage, ToolMessage 모두 누적 |
| `thoughts` | think 노드 (append) | LLM의 content text (도구 호출 시점의 의도 한 줄 또는 최종 답변 본문) |
| `actions` | think 노드 (append) | LLM이 발행한 tool_call dict (`{name, args, id}`) |
| `observations` | observe 노드 (append) | ToolMessage content들 |
| `iteration_count` | think 노드 (+1) | think 호출 횟수. 최종 답변 turn까지 누적 |
| `max_iterations` | 첫 think (= 6 기본) | 도달 시 should_continue가 강제 finish |
| `current_goal` | 첫 think (한 번만) | 사용자 첫 메시지 첫 200자 |
| `final_answer` | finish 노드 | 최종 AIMessage.content 발췌 |
| `is_complete` | finish 노드 | 종료 신호 (외부에서 확인 가능) |

**도메인 컨텍스트**

| 필드 | 채워지는 시점 | 비고 |
|---|---|---|
| `image_path`, `gender` | HumanMessage 파싱 또는 tool_call args | 정규식으로 사용자 입력에서 추출 |
| `skin_scores` | `skin_analyze` 완료 | `{age, age_note, valid_sagging, raw_scores{13개 키}}` |
| `top_concerns` | `skin_analyze` 완료 | 부위별 평균 점수 낮은(=심각한) 순 3개 |
| `db_recommendations` | `recommend_treatment_db` 완료 | region별 dict (feature_name, code, treatment, customer_desc, matched_by 등) |
| `pubmed_recommendations` | `search_pubmed` 완료 | region별 dict (pmid, title, authors, year, study_type, abstract, similarity 등). 점수 낮은 top 3 부위만 |

영속화: `InMemorySaver` + `thread_id`로 turn 간 state 유지. `ChatSession`은 `thread_id` 자동 발급, multi-turn 대화 가능.

---

## pipeline — TFLite 추론

[pipeline/pipeline.py](pipeline/pipeline.py)의 `SkinPipeline` 클래스가 얼굴 이미지 1장을 받아 다음 항목을 0~100 raw 점수로 산출 (낮을수록 심각):

```
predict_single(image_path, gender="female") →
{
  "age":           float,                              # 추정 피부 연령 (성별 의존)
  "pigment":       {left, right},                      # 좌/우 볼 색소
  "wrinkle":       {forehead, right_eye, left_eye, nasolabial, perioral, right_vol, left_vol},
  "homogenity":    {radiance, texture},                # 광채/거칠기
  "cheek_sagging": {right, left, total},               # 볼 처짐
  "chin_sagging":  {right, left, total},               # 턱 처짐
  "valid_sagging": bool                                # 정면 각도 검증
}
```

내부 구성:
- [preprocess.py](pipeline/preprocess.py) — MediaPipe FaceLandmarker로 부위별 crop
- [effnet_predictor.py](pipeline/effnet_predictor.py) — EfficientNet TFLite 추론 (age/pigment/wrinkle/homogenity)
- [sagging_predictor.py](pipeline/sagging_predictor.py) — 별도 모델 + 성별 파라미터로 볼/턱 처짐
- [config.yaml](pipeline/config.yaml) — TFLite 모델 경로, sagging 파라미터, root_dir
- `inference_models/` — `.tflite` 모델 + 성별 파라미터 JSON

**단독 실행**

```bash
python pipeline/pipeline.py --image samples/028_data/0001/0001_01_F.jpg --gender female
```

**에이전트에서 호출**: `tools/skin_analyze.py`가 `SkinPipeline`을 지연 import하고 gender별로 캐싱(TFLite 로딩 비용 회피).

---

## classification — 모델 학습

[classification/](classification/)는 pipeline이 사용하는 TFLite 모델을 만든 학습·추론 코드 모음 (별도 사이클, 런타임에 직접 호출되지 않음):

| 파일/디렉터리 | 역할 |
|---|---|
| `train.py` | EfficientNet 학습 |
| `inference.py` | PyTorch/TF 추론 (변환 전 검증용) |
| `make_tflite_models.py` | 학습된 모델 → `.tflite` 변환 |
| `models/`, `dataloaders/`, `tasks/`, `convert/`, `preprocess/`, `custom_losses/`, `utils/` | 학습 구성요소 |
| `configs/` | 학습 config YAML |
| `confusion_matrix.py`, `age_viz.py` | 평가/시각화 |

산출물(`.tflite`)은 [pipeline/inference_models/](pipeline/inference_models/)로 복사돼 SkinPipeline이 사용한다.

---

## auradb — Neo4j 시술 규칙 DB

**핵심 파일**

| 파일 | 역할 |
|---|---|
| [data/mapping_table_tmp.xlsx](auradb/data/) | 시술 규칙 마스터 (9개 시트 = 카테고리, 110+행) |
| [aura_upload_beauty_rules.py](auradb/aura_upload_beauty_rules.py) | Excel → `FeatureRule` 노드 + 벡터 인덱스로 적재. 단일/2조건/3조건 시트 모두 파싱, customer_desc 임베딩(`text-embedding-3-small`), 적재 전 `wipe_existing()`으로 청소 |
| [Connect_DB.py](auradb/Connect_DB.py) | Neo4j driver + 조회 헬퍼 (`fetch_feature_rows`, `search_feature_by_name`, `list_age_groups`) |
| [rule_matcher.py](auradb/rule_matcher.py) | 결정론적 매칭. 4가지 condition 패턴 (단순 범위, 양쪽 비교, prefixed range, 가중합) 파서 + feature별 dispatcher |
| [add_case.py](auradb/add_case.py) | 환자 케이스(`CombinedRule`) 시뮬레이션 적재 (옵션) |

**스키마**

```
(:FeatureRule {
  rule_id, feature_name, age_group,
  condition_1, condition_2, condition_3,   ← rule_matcher가 파싱
  raw_condition, code, output_text,
  customer_desc, clinician_desc,
  source_sheet, embedding_text, embedding[1536]
})
```

UNIQUE: `(feature_name, age_group, code)`. 벡터 인덱스: `feature_rule_embedding_idx`(cosine).

**카테고리 (= 시트명) 9종**

| feature_name | 코드 prefix | 결정 컬럼 |
|---|---|---|
| 색소 | P_0..P_5 | 연령 + 좌/우 볼 색소 점수 |
| 이마·미간주름 | FW_0..FW_3 | 연령 + 이마주름값 |
| 눈가·앞광대주름 | EW_0..EW_4 | 연령 + 좌/우 눈가 점수 |
| 팔자주름 | NW_0..NW_3 | 연령 + 팔자주름값 |
| 입가·턱주름 | OW_0..OW_3 | 연령 + 입가턱주름값 |
| 리프팅(윤곽) | L_0, C_300, F_*, J_* (11종) | 연령 + 1차/2차/3차 조건(가중합) |
| 리프팅(탄력) | LT_0..LT_3 | 연령 + 가중합 + 턱처짐 |
| 스킨부스터 (new) | SB_0..SB_6 | 연령 + 광채 + 거칠기 |
| 볼꺼짐 | V_0..V_3 | 연령 + 좌/우 볼꺼짐 |

**적재 (재생성)**

```bash
docker exec -w /workspace skin_inference python auradb/aura_upload_beauty_rules.py
```

기존 노드 전체 청소 후 `mapping_table_tmp.xlsx`에서 재적재. OpenAI 임베딩 비용 ~$0.001, 시간 10~30초.

---

## search_pubmed — PubMed RAG 시술 근거

[beauty-agent/tools/search_pubmed.py](beauty-agent/tools/search_pubmed.py)는 진단 결과를 받아 NCBI E-utilities로 시술 관련 논문을 가져오고 OpenAI 임베딩 cosine 유사도로 랭킹한다. auradb와 **독립**적인 evidence 소스.

**Flow per 부위**

1. `aggregate_regions(raw_scores)` → 점수 낮은(=심각한) top 3 부위 선정
2. region 키 → 영문 검색어 매핑 (예: `nasolabial_fold` → `"nasolabial fold filler treatment"`) + severity(`severe`/`moderate`/`mild`) suffix
3. `esearch.fcgi?db=pubmed&sort=relevance&retmax=5` → PMID 후보 5개
4. `efetch.fcgi?rettype=abstract&retmode=xml` → XML 파싱(제목·저자·연도·초록·study_type)
5. query 임베딩 vs 각 article(title+abstract) 임베딩 → cosine 정렬 → top-1 선정
6. `state.pubmed_recommendations`에 region별 결과 dict 적재 + LLM에 자연어 요약 ToolMessage 전달

**비용/성능**

- 부위당 esearch 1 + efetch 1 = PubMed API 2회 (top 3 부위 → 6회)
- 임베딩: query 1 + abstract 5 = 부위당 6회 (top 3 부위 → 18회, `text-embedding-3-small` 기준 < $0.001)
- E-utilities는 API key 없이 ~3 req/sec 제한 (현 코드 충분)

**호출 트리거** (system prompt 기준)

- 사용자가 "논문", "근거", "evidence", "PubMed" 등을 명시했을 때만 호출
- 비명시 시 임의 호출 금지 (외부 API 비용 의식)

**향후 debate 단계**

`db_recommendations`(결정론·규칙 기반)와 `pubmed_recommendations`(증거·RAG 기반)을 동시에 보유하는 시점에서 final report 노드가:
- 두 소스를 부위별로 정렬 + 매칭
- 가중 평균(예: DB 60% : 논문 40%) 또는 가중 prompt로 LLM이 통합 결론 생성
- 출처(PMID, 코드) 명시 포함

이 부분은 현재 미구현. 두 결과만 state에 적재하고 LLM이 turn별로 따로 답변.

---

## docker — CPU 전용 런타임

[docker/Dockerfile](docker/Dockerfile): `python:3.11-slim` 기반. tensorflow-cpu + mediapipe + langchain/langgraph/neo4j/openpyxl 설치. user1(UID 1000)으로 실행, `/workspace` 마운트.

**권장: Docker Compose ([docker-compose.yml](docker-compose.yml))**

```bash
docker compose build                              # 이미지 빌드 (skin_inference:latest)
docker compose run --rm beauty-agent              # 채팅 실행 (python main.py)
docker compose run --rm beauty-agent bash         # 셸 진입
```

**기존 스크립트 (동일 동작)**

```bash
bash docker_build.sh    # 이미지 빌드 (skin_inference:latest)
bash docker_run.sh      # 컨테이너 기동 (-v .:/workspace, name=skin_inference)
bash docker_attach.sh   # 셸 진입
```

둘 다 호스트 CWD를 `/workspace`에 바인드 마운트하므로 코드 수정이 컨테이너에 즉시 반영된다.
Compose는 `.env`를 자동으로 주입(`env_file`)하므로 별도 `-e` 옵션이 필요 없다.

---

## 환경 변수

[.env](.env) (루트 단일 파일, 모든 모듈이 공유. `.gitignore`로 제외)

```
# LLM
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
PUBMED_API_KEY=

# AuraDB
NEO4J_URI=neo4j+s://....databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=

# Embedding
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_EMBED_DIMENSIONS=1536

# Misc
EXCEL_PATH=auradb/data/mapping_table_tmp.xlsx
```

샘플은 [.env.example](.env.example). LLM 모델 ID는 [beauty-agent/config.py](beauty-agent/config.py)의 `AI_MODEL`에서 직접 지정 (예: `openai:gpt-4o-mini`).

---

## 빠른 시작

기준 실행 위치: 프로젝트 루트. **Docker + Docker Compose만 있으면 됩니다.**

```bash
# 1) 저장소 클론
git clone <repo-url>
cd capstone_design_team10

# 2) .env 작성 (루트) — 실제 키 채우기
cp .env.example .env
$EDITOR .env  # OPENAI_API_KEY, NEO4J_* 등

# 3) 모델 가중치 배치 (최초 1회) — 아래 '모델 파일' 섹션 참고
#    pipeline/inference_models/ 아래에 .tflite / .task 파일이 있어야 합니다.

# 4) 이미지 빌드 (최초 1회)
docker compose build

# 5) AuraDB에 시술 규칙 적재 (최초 1회 / 데이터 변경 시)
docker compose run --rm beauty-agent python auradb/aura_upload_beauty_rules.py

# 6) 채팅 실행
docker compose run --rm beauty-agent
```

> 기존 `docker_build.sh` / `docker_run.sh` / `docker_attach.sh` 스크립트도 그대로 동작하지만,
> 신규 사용자는 위 `docker compose` 흐름을 권장합니다.

### 모델 파일 (필수)

`pipeline/config.yaml`이 참조하는 추론 모델은 용량이 커서 Git에 포함되지 않을 수 있습니다.
clone 직후 아래 파일들이 `pipeline/inference_models/` 아래 있는지 확인하세요(없으면 진단이 실패합니다):

```
pipeline/inference_models/face_landmarker.task
pipeline/inference_models/age_reg/*.tflite
pipeline/inference_models/pigment_reg/*.tflite
pipeline/inference_models/wrinkle_reg/*.tflite
pipeline/inference_models/homogenity_reg/*.tflite
pipeline/inference_models/{cheek,chin}-{male,female}_params.json   # (Git에 포함)
```

없으면 모델 제공자에게 별도 전달받아 위 경로에 두세요(또는 Git LFS / 릴리스 첨부로 배포).

채팅 예:
```
You> 이 사진에 대한 시술 추천해줘. 성별은 여자, 경로: samples/028_data/0001/0001_01_F.jpg

Agent>
🔧 [Act] skin_analyze(image_path='samples/...', gender='female')

📋 [Observe]
피부 진단 결과 (점수 0~100, 낮을수록 심각):
- 팔자주름: 44.0점
...

🔧 [Act] recommend_treatment_db()

📋 [Observe]
AuraDB FeatureRule 조회 결과
[추천 시술 - 심각도가 높은 카테고리 3개]
1. 팔자주름 ... NW_3
2. 스킨부스터 ... SB_6
3. 이마·미간주름 ... FW_2

분석 결과 다음과 같은 시술을 추천드립니다:
1. **팔자주름** — 필러/스킨부스터 + 리프팅
2. **스킨부스터** — 얼굴 전체 스킨부스터 (쥬베룩/리쥬란)
3. **이마·미간주름** — 보톡스 + 필러/스킨부스터
```

---

## 데이터 흐름 한 번 더

```
사용자 입력 ("진단해줘 / 추천해줘" + 이미지 경로 + 성별)
      │
      ▼
ChatSession.stream
      │ HumanMessage + image_path/gender 파싱
      ▼
agent_node (LLM bind_tools)
      │ Thought + tool_call 발행
      ▼
skin_analyze_node ──▶ pipeline/SkinPipeline.predict_single ──▶ 13개 raw 점수
      │                                                       state.skin_scores 갱신
      ▼
agent_node (state 갱신 반영)
      │ 추천 요청이면 tool_call 또 발행
      ▼
recommendation_node ──▶ Connect_DB.fetch_feature_rows ──▶ rule_matcher.match_feature_row
      │                                                  (조건 평가, 매칭 실패 시 cosine fallback)
      │                                                  state.db_recommendations 갱신
      ▼
agent_node (final answer)
      │
      ▼
사용자에게 token stream
```

---

## 트러블슈팅

| 증상 | 원인 / 대처 |
|---|---|
| `Failed to DNS resolve ...databases.neo4j.io` | AuraDB free tier 인스턴스 일시정지. 콘솔에서 resume |
| `RESOURCE_EXHAUSTED (429)` (Gemini) | free tier 일일 한도 (20 req/day). 다음날 재시도 또는 `AI_MODEL` 교체 |
| `mapping_table_tmp.xlsx 찾을 수 없음` | `.env`의 `EXCEL_PATH=auradb/data/mapping_table_tmp.xlsx` 확인 |
| 추천 코드가 잘못된 임계 매칭 | rule_matcher가 condition 파싱 실패한 row는 skip. row 텍스트의 오타 확인 (예: "75-10점") |
| `search_pubmed` 호출이 텍스트로만 시뮬레이션됨 | LLM이 function-calling 대신 ```json``` 블록을 본문에 적는 케이스. system prompt 상단의 "도구 호출 방식 (절대 규칙)" 섹션이 이를 막음. 그래도 발생 시 모델 교체(gpt-4o → gpt-4o-mini → claude 등)나 max_tokens 조정 |
| PubMed esearch/efetch 타임아웃 | 컨테이너 → 외부 인터넷 확인. PUBMED_API_KEY를 발급받아 .env에 추가하면 rate limit 완화 (3 → 10 req/sec) |
| 컨테이너에 패키지 없음 | Dockerfile 수정 → `bash docker_build.sh` 재빌드 → `docker stop skin_inference; bash docker_run.sh` |

---

## 향후 확장

- **Final report 노드 + DB-PubMed debate** — "레포트 작성해줘" trigger 감지 → 별도 노드로 분기.
  `db_recommendations`와 `pubmed_recommendations` 두 소스를 부위별 매칭 + 가중치(예: 6:4)로 통합해
  최종 시술 추천 보고서 생성. 인용 출처(PMID + DB code) 동봉.
- **상담 챗봇 메모리** — InMemorySaver → SQLite/Postgres 체크포인터로 멀티유저·영속화
- **PubMed RAG 강화** — abstract을 청크로 쪼개 sentence-level RAG, MeSH term 활용, 부위별 top-K 확장
