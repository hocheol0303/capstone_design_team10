from PIL import Image
import numpy as np
from preprocess.crop_data import crop_cheeks_from_detection_result, crop_left_cheek_from_detection_result, crop_right_cheek_from_detection_result
from utils.Mediapipe import MediaPipe

face_landmarker = None

def init_worker():
    global face_landmarker
    face_landmarker = MediaPipe()

def save_image(img: np.ndarray, path: str):
    img = img.astype('uint8')
    Image.fromarray(img).save(path)

def process_one(item):
    global face_landmarker
    source_path, left_path, right_path = item
    try:
        with open(source_path, 'rb') as f:
            image_contents = f.read()
        
        mp_image = face_landmarker.create_mp_image(image_contents)

        detection_result = face_landmarker.landmarker.detect(mp_image)
        if not detection_result.face_landmarks:
            print(f"No face detected in {source_path}")
            return source_path
        
        landmarks = detection_result.face_landmarks[0]
        left_cheek, right_cheek = crop_cheeks_from_detection_result(landmarks, mp_image)

        save_image(left_cheek, left_path)
        save_image(right_cheek, right_path)
        return None
    except Exception as e:
        print(f"Error processing {source_path}: {e}")
        return source_path

def process_left(item):
    global face_landmarker
    source_path, left_path = item
    try:
        with open(source_path, 'rb') as f:
            image_contents = f.read()
        mp_image = face_landmarker.create_mp_image(image_contents)
        
        detection_result = face_landmarker.landmarker.detect(mp_image)
        if not detection_result.face_landmarks:
            print(f"No face detected in {source_path}")
            return source_path
        
        landmarks = detection_result.face_landmarks[0]
        left_cheek = crop_left_cheek_from_detection_result(landmarks, mp_image)

        save_image(left_cheek, left_path)
        return None
    except Exception as e:
        print(f"Error processing {source_path}: {e}")
        return source_path

def process_right(item):
    global face_landmarker
    source_path, right_path = item
    try:
        with open(source_path, 'rb') as f:
            image_contents = f.read()
        mp_image = face_landmarker.create_mp_image(image_contents)
        
        detection_result = face_landmarker.landmarker.detect(mp_image)
        if not detection_result.face_landmarks:
            print(f"No face detected in {source_path}")
            return source_path
        
        landmarks = detection_result.face_landmarks[0]
        right_cheek = crop_right_cheek_from_detection_result(landmarks, mp_image)

        save_image(right_cheek, right_path)
        return None
    except Exception as e:
        print(f"Error processing {source_path}: {e}")
        return source_path