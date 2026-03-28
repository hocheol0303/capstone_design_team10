import cv2
from utils.Mediapipe import MediaPipe
import numpy as np
import multiprocessing
from tqdm import tqdm
import pandas as pd
import os
import re

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


# 전역 변수 선언(각 프로세스에서 개별 로드 됨)
detector = None

def worker_init():
    '''워커 프로세스가 시작될 때 MediaPipe 인스턴스를 한 번만 생성'''
    global detector
    detector = MediaPipe()

def get_hist(task):
    image_path, dataset = task

    try:
        with open(image_path, 'rb') as f:
            img_byte = f.read()
        
        mp_image = detector.create_mp_image(img_byte)
        detection_result = detector.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return {'path': image_path[image_path.find(dataset):], 'status': 'fail_no_face'}
        
        landmarks = detection_result.face_landmarks[0]
        np_image = mp_image.numpy_view()
        h, w, _ = np_image.shape

        # FACE_OVAL 좌표 추출
        face_coords = np.array([
            [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in FACE_OVAL
        ], np.int32)

        # 이미지 크기와 동일한 검은 화면 생성 후 얼굴 영역을 흰색으로 채우기(boolean mask로 얼굴 영역의 픽셀만 추출하기 위해서)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_coords], 255)

        hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
        v_channel = hsv_image[:, :, 2]
        face_pixels = v_channel[mask == 255]

        if len(face_pixels) == 0:
            return {'path': image_path[image_path.find(dataset):], 'status': 'fail_no_face_pixels'}
        
        # 지표 계산
        v_mean = np.mean(face_pixels)
        v_std = np.std(face_pixels)
        low_ratio = np.sum(face_pixels < 60) / len(face_pixels)
        high_ratio = np.sum(face_pixels > 200) / len(face_pixels)

        return {
            'path': image_path[image_path.find(dataset):],
            'status': 'success',
            'v_mean': v_mean,
            'v_std': v_std,
            'low_ratio': low_ratio,
            'high_ratio': high_ratio
        }

    except Exception as e:
        return {'path': image_path[image_path.find(dataset):], 'status': f'error: {str(e)}'}

def main():
    # image_paths에 데이터셋 경로 모으기
    dataset_root = '/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data'
    dataset_paths = []
    tasks = []
    for dataset in os.listdir(dataset_root):
        if re.search(r"[0-9]{3}_data", dataset):
            dataset_paths.append(os.path.join(dataset_root, dataset))
    
    for dataset_path in dataset_paths:
        dataset = dataset_path.split('/')[-1]
        for root, _, files in os.walk(os.path.join(dataset_path, 'data')):
            for file in files:
                # multiprocessing pool에 던질 수 있도록 평탄화(1차원 리스트 까서 보고 바로 처리할 수 있도록)
                tasks.append((os.path.join(root, file), dataset))
    print(f"총 이미지 개수: {len(tasks)}")

    # 멀티프로세싱
    num_workers = multiprocessing.cpu_count() // 2

    with multiprocessing.Pool(num_workers, initializer=worker_init) as pool:
        results = list(tqdm(pool.imap(get_hist, tasks), total=len(tasks)))
    
    success_results = []
    fail_results = []

    for result in results:
        if result['status'] == 'success':
            success_results.append(result)
        else:
            fail_results.append(result)
    
    success_df = pd.DataFrame(success_results)
    fail_df = pd.DataFrame(fail_results)

    success_df.to_csv('/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/hist_success_results.csv', index=False)
    fail_df.to_csv('/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/hist_fail_results.csv', index=False)

if __name__ == '__main__':
    main()