import os
import uuid
from pathlib import Path

from neo4j import GraphDatabase

##import pandas as pd


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env_file(Path(__file__).resolve().parent / ".env")

URI = os.getenv("NEO4J_URI", "")
USERNAME = os.getenv("NEO4J_USERNAME", "c995342e")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


# ---------------- 저장 ----------------
def save_face(tx, features):
    tx.run(
        """
    CREATE (u:UserFace {
        id: $id,
        age: $age,
        pigment: $pigment,
        wrinkle_forehead: $wf,
        wrinkle_eye: $we,
        wrinkle_smile: $ws,
        lifting_contour: $lc,
        lifting_elasticity: $le,
        embedding: $embedding
    })
    """,
        id=str(uuid.uuid4()),
        age=features[0],
        pigment=features[1],
        wf=features[2],
        we=features[3],
        ws=features[4],
        lc=features[5],
        le=features[6],
        embedding=features,
    )


# ---------------- 유사도 검색 ----------------
def find_similar(tx, features):
    result = tx.run(
        """
    CALL db.index.vector.queryNodes(
      'face_embedding_index',
      3,
      $embedding
    )
    YIELD node, score
    RETURN node, score
    """,
        embedding=features,
    )

    return [record for record in result]


# ---------------- 룰 기반 추천 ----------------
def rule_based_recommend(features):
    age, pigment, wf, we, ws, lc, le = features

    recommendations = []

    if pigment > 0.7:
        recommendations.append("미백 케어")

    if wf > 0.5 or we > 0.5 or ws > 0.5:
        recommendations.append("주름 개선 / 리프팅")

    if lc < 0.4:
        recommendations.append("윤곽 리프팅")

    return recommendations


# ---------------- 메인 실행 ----------------
def main():
    features = [28, 0.6, 0.3, 0.2, 0.4, 0.5, 0.6]

    with driver.session() as session:
        session.execute_write(save_face, features)

    with driver.session() as session:
        result = session.execute_read(find_similar, features)

    print("=== 유사 결과 ===")
    for r in result:
        print(dict(r["node"]), r["score"])


# 실행
if __name__ == "__main__":
    main()
