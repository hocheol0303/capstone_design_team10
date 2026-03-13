import io
import tempfile
from pathlib import Path

from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


class MediaPipe:
    def __init__(self):
        """FaceLandmarker 인스턴스를 초기화합니다."""
        base_options = python.BaseOptions(model_asset_path="/home/hocheol/inskin_ai/EfficientNet/src/face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def create_mp_image(self, content: bytes):
        temp_path = Path(tempfile.gettempdir()) / f"temp_{id(content)}.png"
        try:
            image = Image.open(io.BytesIO(content))
            image = ImageOps.exif_transpose(image)
            image.save(temp_path, "PNG")

            mp_image = mp.Image.create_from_file(str(temp_path))
            return mp_image
        finally:
            if temp_path.exists():
                temp_path.unlink()


def get_mp_image(face_landmarker, file_path):
    with open(file_path, 'rb') as f:
        img_byte = f.read()
    mp_image = face_landmarker.create_mp_image(img_byte)

    return mp_image

def get_landmarks(face_landmarker, mp_image):
    detection_result = face_landmarker.landmarker.detect(mp_image)

    if not detection_result.face_landmarks:
        return None
    else:
        landmarks = detection_result.face_landmarks[0]
        np_image = mp_image.numpy_view()
        return landmarks, np_image

def landmarks_scaling(landmarks, height, width):
    '''
    landmarks의 x, y 좌표를 이미지 크기에 맞게 스케일링 합니다.
    '''
    for idx, lm in enumerate(landmarks):
        lm.x = int(lm.x * width)
        lm.y = int(lm.y * height)
    
    return landmarks

def process(face_landmarker, file_path):
    '''
    return: np_image, scaled_landmarks, face_boxes
    '''
    mp_image = get_mp_image(face_landmarker, file_path)
    landmarks, np_image = get_landmarks(face_landmarker, mp_image)
    if landmarks is None:
        raise Exception("얼굴이 감지되지 않았습니다.")
    h, w = np_image.shape[:2]
    scaled_landmarks = landmarks_scaling(landmarks, h, w)

    # 얼굴 영역 계산
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for lm in scaled_landmarks:
        min_x = min(min_x, lm.x)
        min_y = min(min_y, lm.y)
        max_x = max(max_x, lm.x)
        max_y = max(max_y, lm.y)
    face_boxes = (min_x, min_y, max_x, max_y)

    return np_image, scaled_landmarks, face_boxes