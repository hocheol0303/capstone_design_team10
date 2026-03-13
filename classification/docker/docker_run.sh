# --gpus all: 모든 GPU를 할당
# -it: interactive 모드로 실행하며 터미널에 연결(in/out 연결)
# -e TZ=Asia/Seoul: 환경변수 TZ를 설정해서 시간대를 서울로 맞춤
# --name: 컨테이너 이름 지정
# -v [local path]: [container path]: local path와 container path를 연결
# -w: 컨테이너 시작될 때 들어갈 디렉터리
# hobbang: 사용할 Docker 이미지 이름(docker build한 이미지)
# -d: detached 모드로 실행하며 백그라운드에서 실행
# tail -f /dev/null: 컨테이너가 종료되지 않도록 유지하기 위해 /dev/null을 tail로 계속 읽음

docker run --rm -dit --gpus all \
        --shm-size=32gb \
        -v /home/hocheol/inskin_ai:/home/hocheol/inskin_ai \
        --name hobbang_effnet \
        -e HOME=/home/hocheol/inskin_ai \
        hobbang_effnet:latest
