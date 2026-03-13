import sys
sys.path.append('/home/work/hocheol_dir/workspace/_dataset')
from preprocess.remove_non_skin import remove_non_skin
from utils.Mediapipe import MediaPipe
from preprocess.pigment_data.crop_mp import crop_cheeks_from_np_image, crop_left_cheek_from_np_image, crop_right_cheek_from_np_image
from PIL import Image
import os
# import matplotlib.pyplot as plt
# from tqdm import tqdm
import re
from multiprocessing import current_process

face_landmarker = None

def initializer():
    global face_landmarker
    print(f"\033[41mInitializing MediaPipe for process: {current_process().name}\033[0m")
    face_landmarker = MediaPipe()

def process_image(img_path):
    global face_landmarker
    target_root = '/home/work/hocheol_dir/workspace/_dataset/datasets/pigment_data/v0.4/v0.4_data'
    origin_root = '/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data'
    try:
        origin_path = os.path.join(origin_root, img_path)
        target_path = os.path.join(target_root, img_path)
    
        left_path = re.sub(r"_data/data/", r"_data/data/left_", target_path)
        # left_path = os.path.splitext(left_path)[0] + ".jpg"

        right_path = re.sub(r"_data/data/", r"_data/data/right_", target_path)
        # right_path = os.path.splitext(right_path)[0] + ".jpg"

        if os.path.exists(left_path) and os.path.exists(right_path):
            return f"skipped: {img_path}"
        
        with open(origin_path, 'rb') as f:
            img_bytes = f.read()
            mp_image = face_landmarker.create_mp_image(img_bytes)
        
        np_image = mp_image.numpy_view()
        detection_results = face_landmarker.landmarker.detect(mp_image)

        if not detection_results.face_landmarks:
            return f"failed (no landmarks): {img_path}"
        
        landmarks = detection_results.face_landmarks[0]
        masked_image = remove_non_skin(np_image, landmarks)

        left_cheek = crop_left_cheek_from_np_image(masked_image, landmarks)
        right_cheek = crop_right_cheek_from_np_image(masked_image, landmarks)

        results = ''

        try:
            if 0 not in left_cheek.shape:
                os.makedirs(os.path.dirname(left_path), exist_ok=True)
                Image.fromarray(left_cheek.astype('uint8')).save(left_path)
        except Exception as e:
            results += f"failed: {img_path}\n"

        try:
            if 0 not in right_cheek.shape:
                os.makedirs(os.path.dirname(right_path), exist_ok=True)
                Image.fromarray(right_cheek.astype('uint8')).save(right_path)
        except Exception as e:
            results += f"failed: {img_path}"

        return results
    
    except Exception as e:
        # send_slack_message(f"Error processing {img_path}: {e}")
        return f"failed (exception): {img_path} - {e}"
    
# 092 전용 버전
def process_image2(filename):
    global face_landmarker
    target_root = '/home/work/hocheol_dir/workspace/_dataset/datasets/pigment_data/v0.4/v0.4_data'
    source_root = '/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/092_data/data'
    
    image_path = os.path.join(source_root, filename)

    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        mp_image = face_landmarker.create_mp_image(image_bytes)
        detection_results = face_landmarker.landmarker.detect(mp_image)

        if not detection_results.face_landmarks:
            return f"failed (no landmarks): {image_path}"
        
        landmarks = detection_results.face_landmarks[0]
        np_image = mp_image.numpy_view()

        masked_image = remove_non_skin(np_image, landmarks)

        left_cheek = crop_left_cheek_from_np_image(masked_image, landmarks)
        right_cheek = crop_right_cheek_from_np_image(masked_image, landmarks)

        results = ''

        for sector, img in zip(['left', 'right'], [left_cheek, right_cheek]):
            if 0 not in img.shape:
                sector_path = os.path.join('092_data', 'test_data', f"{sector}_{filename}")
                os.makedirs(os.path.dirname(os.path.join(target_root, sector_path)), exist_ok=True)
                Image.fromarray(img.astype('uint8')).save(os.path.join(target_root, sector_path))

                results += f"success: {sector_path}\n"
    except Exception as e:
        return f"failed: {image_path}, {str(e)}\n"
    
    return results