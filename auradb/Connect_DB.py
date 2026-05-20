import os
from pathlib import Path

from neo4j import GraphDatabase
from openai import OpenAI


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env_file(Path(__file__).resolve().parent.parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_EMBED_DIMENSIONS = int(os.getenv("OPENAI_EMBED_DIMENSIONS", "1536"))

openai_client = OpenAI()
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def _session():
    if NEO4J_DATABASE:
        return driver.session(database=NEO4J_DATABASE)
    return driver.session()


def embed(text: str) -> list:
    response = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text,
        dimensions=OPENAI_EMBED_DIMENSIONS,
    )
    return response.data[0].embedding


def search_feature(query_text: str, top_k: int = 1) -> list[dict]:
    """자연어 텍스트로 가장 유사한 FeatureRule을 검색하고 output_text를 반환."""
    vector = embed(query_text)
    with _session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('feature_rule_embedding_idx', $top_k, $vector)
            YIELD node, score
            RETURN node.feature_name AS feature_name,
                   node.output_text  AS output_text,
                   node.customer_desc AS customer_desc,
                   score
            """,
            top_k=top_k,
            vector=vector,
        )
        return [dict(r) for r in result]


def _cosine(a: list, b: list) -> float:
    """두 벡터의 코사인 유사도. 입력이 비어 있으면 0."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search_feature_by_name(
    feature_name: str,
    query_text: str,
    age_group: str | None = None,
    top_k: int = 1,
) -> list[dict]:
    """feature_name(+age_group) 제약 하에 코사인 유사도 검색.

    벡터 인덱스(전역 top-K)는 임베딩 품질이 나쁜 feature가 누락될 수 있어,
    feature_name으로 먼저 좁힌 뒤 Python에서 cosine을 계산한다.
    각 feature의 행 수는 십수 개 수준이라 비용 차이는 무시할 만하다.
    """
    vector = embed(query_text)
    with _session() as session:
        rows = session.run(
            """
            MATCH (n:FeatureRule {feature_name: $feature_name})
            WHERE $age_group IS NULL OR n.age_group = $age_group
            RETURN n.feature_name  AS feature_name,
                   n.age_group     AS age_group,
                   n.code          AS code,
                   n.output_text   AS output_text,
                   n.customer_desc AS customer_desc,
                   n.clinician_desc AS clinician_desc,
                   n.embedding     AS embedding
            """,
            feature_name=feature_name,
            age_group=age_group,
        ).data()

    if not rows:
        return []

    scored = []
    for r in rows:
        emb = r.pop("embedding", None)
        r["score"] = _cosine(vector, emb) if emb else 0.0
        scored.append(r)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def fetch_feature_rows(feature_name: str, age_group: str | None = None) -> list[dict]:
    """feature_name(+age_group) 제약 하에 모든 row를 반환 (벡터 검색 없음).

    rule-based 매칭에 필요한 condition_1/2/3, code, output_text, customer_desc 등을 포함.
    embedding 컬럼은 제외 (대용량 + rule_matcher에서 불필요).
    """
    with _session() as session:
        result = session.run(
            """
            MATCH (n:FeatureRule {feature_name: $feature_name})
            WHERE $age_group IS NULL OR n.age_group = $age_group
            RETURN n.feature_name   AS feature_name,
                   n.age_group      AS age_group,
                   n.condition_1    AS condition_1,
                   n.condition_2    AS condition_2,
                   n.condition_3    AS condition_3,
                   n.code           AS code,
                   n.output_text    AS output_text,
                   n.customer_desc  AS customer_desc,
                   n.clinician_desc AS clinician_desc
            """,
            feature_name=feature_name,
            age_group=age_group,
        )
        return [dict(r) for r in result]


def list_age_groups(feature_name: str) -> list[str]:
    """해당 feature_name의 distinct age_group 목록."""
    with _session() as session:
        result = session.run(
            """
            MATCH (n:FeatureRule {feature_name: $feature_name})
            RETURN DISTINCT n.age_group AS age_group
            """,
            feature_name=feature_name,
        )
        return [r["age_group"] for r in result if r["age_group"]]


def recommend(vlm_output: str) -> str:
    """
    VLM 출력 문자열을 받아 특성별로 분리 후 각각 벡터 검색하여
    시술 추천 결과를 '; '로 연결해 반환.

    vlm_output 형식 (;로 특성 구분, 각 블록은 '특성명: 설명' 형태):
        색소: 얼굴 전반에서 색소 병변이 보입니다.\n 5회 정도 색소 치료가 필요해 보입니다.; 이마·미간주름: ...
    """
    results = []

    for block in vlm_output.split(";"):
        block = block.strip()
        if not block or ":" not in block:
            continue

        feature_name, desc = block.split(":", 1)
        feature_name = feature_name.strip()
        desc = desc.strip()

        if not desc:
            continue

        matches = search_feature(desc, top_k=1)
        if matches:
            output_text = matches[0]["output_text"]
            print(f"  [{feature_name}] → {output_text}  (score: {matches[0]['score']:.4f})")
            results.append(output_text)

    return "; ".join(results)


def main():
    vlm_output = """색소: 얼굴 전반에서 색소 병변이 보입니다.
 5회 정도 색소 치료가 필요해 보입니다.; 이마·미간주름: 이마·미간에 주름이 보입니다.
 보톡스 치료가 필요합니다.; 눈가앞광대주름: 눈가에 주름이 보이기 시작합니다.
 보톡스 치료가 필요합니다.; 팔자주름: 팔자 주름이 보입니다.
 파인 주름을 개선할 필러나 스킨부스터가 필요합니다.; 입가·턱주름: 입주변과 턱에 주름이 보입니다.
 보톡스, 필러, 스킨부스터 등으로 치료가 필요합니다.; 리프팅(윤곽): 얼굴 전체에 처짐이 보입니다.
 피부 처짐을 개선해주는 시술(예를 들어 울쎄라 400샷)이 필요합니다.; 리프팅(탄력): 피부에 탄력 저하가 보입니다.
 피부 타이트닝을 개선해주는 리프팅(예를 들어 써마지 300샷)이 필요합니다.; 볼꺼짐: 볼꺼짐이 보입니다.
 볼꺼짐을 개선할 필러나 스킨부스터가 필요합니다."""

    print("=== 입력 ===")
    print(vlm_output)
    print("\n=== 특성별 검색 결과 ===")
    output = recommend(vlm_output)
    print("\n=== 최종 출력 ===")
    print(output)

    driver.close()


if __name__ == "__main__":
    main()
