from utils.Mediapipe import MediaPipe
from preprocess.remove_non_skin import remove_non_skin
import sys
sys.path.append('/home/hocheol/inskin_ai/EfficientNet/src/preprocess/wrinkle/v0.3/v0.3.0')
from wrinkle_preprocess import wrinkle_preprocess
from utils.constants import SECTORS

import os
from PIL import Image
from multiprocessing import current_process

face_landmarker = None

def initializer():
    global face_landmarker
    print(f"\033[41mInitializing MediaPipe for process: {current_process().name}\033[0m")
    face_landmarker = MediaPipe()

def process_wrinkle_image(filename):
    global face_landmarker
    target_root = '/home/hocheol/inskin_ai/no_track/datasets/wrinkle_data/v0.3/v0.3_data/034_data'
    origin_root = '/home/hocheol/inskin_ai/no_track/datasets/origin_data-simple/034_data'
    try:
        user_id = filename.split('_')[0]
        origin_path = os.path.join(origin_root, user_id, filename)
        


        forehead_path = os.path.join(target_root, user_id, filename.replace(filename, f"{SECTORS[0]}_{filename}"))
        right_eye_path = os.path.join(target_root, user_id, filename.replace(filename, f"{SECTORS[1]}_{filename}"))
        left_eye_path = os.path.join(target_root, user_id, filename.replace(filename, f"{SECTORS[2]}_{filename}"))
        nasolabial_path = os.path.join(target_root, user_id, filename.replace(filename, f"{SECTORS[3]}_{filename}"))
        perioral_path = os.path.join(target_root, user_id, filename.replace(filename, f"{SECTORS[4]}_{filename}"))
        right_vol_path = os.path.join(target_root, user_id, filename.replace(filename, f"{SECTORS[5]}_{filename}"))
        left_vol_path = os.path.join(target_root, user_id, filename.replace(filename, f"{SECTORS[6]}_{filename}"))


        # .JPG로 저장된 이미지 있음
        if not os.path.exists(origin_path):
            ext = origin_path.split('.')[-1]
            origin_path = origin_path.replace(ext, ext.upper())

        with open(origin_path, 'rb') as f:
            img_bytes = f.read()

        mp_image = face_landmarker.create_mp_image(img_bytes)
        detection_results = face_landmarker.landmarker.detect(mp_image)

        if not detection_results.face_landmarks:
            return f"failed (no landmarks): {filename}"
        
        landmarks = detection_results.face_landmarks[0]
        
        np_image = mp_image.numpy_view()
        masked_image = remove_non_skin(np_image, landmarks)

        forehead, right_eye, left_eye, nasolabial, perioral, right_vol, left_vol = wrinkle_preprocess(masked_image, landmarks)

        results = ''

        try:
            if 0 not in forehead.shape:
                if not os.path.exists(forehead_path):
                    os.makedirs(os.path.dirname(forehead_path), exist_ok=True)
                    Image.fromarray(forehead.astype('uint8')).save(forehead_path)
                    results += f"success: {forehead_path}\n"
                else:
                    results += f"exists: {forehead_path}\n"
            if 0 not in right_eye.shape:
                if not os.path.exists(right_eye_path):
                    os.makedirs(os.path.dirname(right_eye_path), exist_ok=True)
                    Image.fromarray(right_eye.astype('uint8')).save(right_eye_path)
                    results += f"success: {right_eye_path}\n"
                else:
                    results += f"exists: {right_eye_path}\n"
            if 0 not in left_eye.shape:
                if not os.path.exists(left_eye_path):
                    os.makedirs(os.path.dirname(left_eye_path), exist_ok=True)
                    Image.fromarray(left_eye.astype('uint8')).save(left_eye_path)
                    results += f"success: {left_eye_path}\n"
                else:
                    results += f"exists: {left_eye_path}\n"
            if 0 not in nasolabial.shape:
                if not os.path.exists(nasolabial_path):
                    os.makedirs(os.path.dirname(nasolabial_path), exist_ok=True)
                    Image.fromarray(nasolabial.astype('uint8')).save(nasolabial_path)
                    results += f"success: {nasolabial_path}\n"
                else:
                    results += f"exists: {nasolabial_path}\n"
            if 0 not in perioral.shape:
                if not os.path.exists(perioral_path):
                    os.makedirs(os.path.dirname(perioral_path), exist_ok=True)
                    Image.fromarray(perioral.astype('uint8')).save(perioral_path)
                    results += f"success: {perioral_path}\n"
                else:
                    results += f"exists: {perioral_path}\n"
            if 0 not in right_vol.shape:
                if not os.path.exists(right_vol_path):
                    os.makedirs(os.path.dirname(right_vol_path), exist_ok=True)
                    Image.fromarray(right_vol.astype('uint8')).save(right_vol_path)
                    results += f"success: {right_vol_path}\n"
                else:
                    results += f"exists: {right_vol_path}\n"
            if 0 not in left_vol.shape:
                if not os.path.exists(left_vol_path):
                    os.makedirs(os.path.dirname(left_vol_path), exist_ok=True)
                    Image.fromarray(left_vol.astype('uint8')).save(left_vol_path)
                    results += f"success: {left_vol_path}\n"
                else:
                    results += f"exists: {left_vol_path}\n"

            return results
        except Exception as e:
            return f"failed: {filename}, {str(e)}\n"
    except Exception as e:
        return f"failed: {filename}, {str(e)}\n"
    

# 092 전용 버전
def process_wrinkle_image2(filename):
    global face_landmarker
    target_root = '/home/work/hocheol_dir/workspace/_dataset/datasets/wrinkle_data/v0.3/v0.3_data'
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

        forehead, right_eye, left_eye, nasolabial, perioral, right_vol, left_vol = wrinkle_preprocess(masked_image, landmarks)

        results = ''

        for sector, img in zip(SECTORS, [forehead, right_eye, left_eye, nasolabial, perioral, right_vol, left_vol]):
            if 0 not in img.shape:
                sector_path = os.path.join('092_data', 'test_data', f"{sector}_{filename}")
                os.makedirs(os.path.dirname(os.path.join(target_root, sector_path)), exist_ok=True)
                Image.fromarray(img).save(os.path.join(target_root, sector_path))
                
                results += f"success: {sector_path}\n"

    except Exception as e:
        return f"failed: {image_path}, {str(e)}\n"
    
    return results