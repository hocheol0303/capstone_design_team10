# 베이스 이미지를 tensorflow 2.15 GPU 버전으로 설정
FROM tensorflow/tensorflow:2.15.0-gpu

# 환경변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/home/hocheol/inskin_ai/.local/bin:${PATH}"

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    tzdata \
    tmux \
    vim \
    zip \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /home/hocheol/inskin_ai

# 의존성 파일 복사, 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    keras==3.4.1 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    opencv-python-headless==4.10.0.84 \
    pillow==12.1.0 \
    matplotlib==3.10.8 \
    scikit-learn==1.8.0 \
    seaborn==0.13.2 \
    tqdm==4.67.1 \
    pyyaml==6.0.3 \
    wandb==0.24.0 \
    mediapipe==0.10.15 \
    ipykernel==7.1.0

RUN ln -sf /home/hocheol/inskin_ai/no_track/.ssh /root/.ssh

# 컨테이너가 뜰 때 커널을 등록하도록 설정
# RUN python -m ipykernel install --name python3 --display-name "Python 3 (Docker)"

CMD ["bash"]

ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} hpcmate || true \
 && useradd  -m -u ${UID} -g ${GID} -s /bin/bash hpcmate || true