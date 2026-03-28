# Dockerfile 찾아서 알아서 빌드
# build: Dockerfile을 기반으로 이미지를 빌드
# -t(--tag) tf_image: 생성되는 이미지에 tf_image라는 태그를 붙임.
# .: 현재 디렉터리에 있는 Dockerfile을 사용해서 이미지를 빌드
docker build -t skin_inference:latest ./docker

