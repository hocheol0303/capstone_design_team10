# CPU 전용 경량 이미지 (pipeline/tflite 추론 전용)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_CACHE_DIR=/workspace/.cache/pip

# OpenCV / MediaPipe 런타임 의존성만 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pipeline 실행에 필요한 패키지만 설치
# tensorflow-cpu: tf.lite.Interpreter 제공 (GPU 드라이버 불필요)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow-cpu==2.15.0 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    opencv-python-headless==4.10.0.84 \
    pyyaml==6.0.3 \
    tqdm==4.67.1 \
    mediapipe==0.10.15 \
    ipykernel==7.1.0 \
    jupyter-server

# beauty-agent (VLM ReAct agent) 의존성
RUN pip install --no-cache-dir \
    langchain>=0.3.0 \
    langchain-google-genai>=2.0.0 \
    langchain-openai>=0.1.0 \
    langgraph>=0.2.0 \
    python-dotenv>=1.0.0 \
    requests>=2.31.0 \
    pydantic>=2.0.0 \
    neo4j>=5.0.0 \
    openpyxl>=3.1.0

# 사용자 그룹과 유저 생성
# -m: 홈 디렉토리 생성, -u: 특정 UID 지정
RUN groupadd -g 1000 user1 && useradd -m -u 1000 -g user1 -s /bin/bash user1

# 작업 디렉토리 설정
WORKDIR /workspace

RUN chown user1:user1 /workspace

USER user1

CMD ["bash"]