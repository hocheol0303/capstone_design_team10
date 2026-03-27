# CPU 전용 경량 이미지 (pipeline/tflite 추론 전용)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# OpenCV / MediaPipe 런타임 의존성만 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /home/hocheol/inskin_ai/pipeline

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
    mediapipe==0.10.15

CMD ["bash"]