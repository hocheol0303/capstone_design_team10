import os
import sys
import json
import logging
from tqdm import tqdm
from PIL import Image
sys.path.append('/home/hocheol/inskin_ai/trashlab')
from parallel_utils import hobbang_multiprocess
import utils.Mediapipe as mp
from preprocess.face_crop_roll import age_preprocess

ORIGIN_ROOT = '/home/hocheol/inskin_ai/no_track/datasets/origin_data-simple/114_data'
FILTERED_ROOT = '/home/hocheol/inskin_ai/no_track/datasets/origin_data/114_data/114_filtered'
REMAINED_ROOT = '/home/hocheol/inskin_ai/no_track/datasets/age_data/v0.3/v0.3_data/114_remained'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 각 워커 프로세스에서 전역으로 사용할 수 있도록 face_landmarker를 초기화
face_landmarker = None

def init_worker():
    """
    각 워커 프로세스가 시작될 때 딱 한 번 실행되어 
    FaceLandmarker를 초기화합니다.
    """
    global face_landmarker
    face_landmarker = mp.MediaPipe()

@hobbang_multiprocess(max_workers=32, initializer=init_worker)
def preprocess_image(src_path, dst_path):
    try:
        # 1. 디렉토리 생성
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # 2. 전처리
        mp_image = mp.get_mp_image(face_landmarker, src_path)
        result = mp.get_landmarks(face_landmarker, mp_image)
        if result is None:
            return False
        landmarks, np_image = result
        preprocessed_image = age_preprocess(landmarks, mp_image)
        Image.fromarray(preprocessed_image.astype('uint8')).save(dst_path)
        return True
    except Exception as e:
        logging.error(f"Error processing {src_path}: {e}")
        return False

if __name__ == "__main__":
    path_pairs = []
    filtered_paths = []
    existing_count = 0

    # 114_filtered에 이미 존재하는 user id 수집    
    label_root = '/home/hocheol/inskin_ai/no_track/datasets/age_data/v0.3/v0.3.0/114_filtered/label'
    data_dict = {}
    for phase in ['train', 'val', 'test']:
        with open(os.path.join(label_root, f"age_{phase}.json"), 'r') as f:
            data_dict[phase] = json.load(f)
    id_sets = {'train': set(), 'val': set(), 'test': set()}
    for phase in ['train', 'val', 'test']:
        for item in data_dict[phase]:
            id_sets[phase].add(item['user_id'])
    merged_ids = id_sets['train'] | id_sets['val'] | id_sets['test']

    logging.info('Scanning files...')
    # 안면인식데이터(114_data)에서 나이 라벨이 있는 user_id 중 114_filtered에 존재하지 않는 path_pairs에 저장
    for user_id in tqdm(merged_ids):
        user_dir = os.path.join(ORIGIN_ROOT, user_id)
        for root, dirs, files in os.walk(user_dir):
            for file in files:
                if file.endswith('.json'):
                    continue
                filtered_path = os.path.join(FILTERED_ROOT, file)
                origin_file_path = os.path.join(root, file)

                remained_file_path = os.path.join(REMAINED_ROOT, user_id, file)
                if not os.path.exists(filtered_path):
                    path_pairs.append((origin_file_path, remained_file_path))
                else:
                    filtered_paths.append(filtered_path)

    logging.info(f"New tasks: {len(path_pairs)}, existing skip: {len(filtered_paths)}")

    # 프로세싱 실행
    if path_pairs:
        summary = preprocess_image(path_pairs)
        print(f"\n[결과 요약] 성공: {summary.get('complete', 0)}, 실패: {summary.get('fail', 0)}")