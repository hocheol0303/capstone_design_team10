from datetime import datetime
start = datetime.now()

import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from utils.constants import SECTORS, SECTOR2IDX
import sys

from utils.Mediapipe import MediaPipe
from preprocess.remove_non_skin import remove_non_skin
sys.path.append('/home/hocheol/inskin_ai/EfficientNet/src/preprocess/wrinkle/v0.3/v0.3.0')
from wrinkle_preprocess import wrinkle_preprocess
# from PIL import Image

'''
tflite로 저장할 당시의 input size로 지정해야함

EfficientNetV2-B0 : (224, 224)
EfficientNetV2-B1 : (240, 240)
EfficientNetV2-B2 : (260, 260)
EfficientNetV2-B3 : (300, 300)
EfficientNetV2-S : (384, 384)
EfficientNetV2-M : (480, 480)
EfficientNetV2-L : (480, 480)
EfficientNetV2-XL (비공식) : (512, 512)

SECTOR:
    forehead
    right_eye
    left_eye
    nasolabial
    perioral
    right_vol
    left_vol
'''

TFLITE_PATH = '/home/hocheol/inskin_ai/EfficientNet/src/convert/results/wrinkle/epoch70_260220_014632.tflite'
IMAGE_PATH = '/home/hocheol/inskin_ai/no_track/datasets/origin_data-simple/092_data/01a6c69e76d729863380b8391d780e53070cf8629ad9ffd5d06826551271fe74/01a6c69e76d729863380b8391d780e53070cf8629ad9ffd5d06826551271fe74_남_30_중립_숙박 및 거주공간_20201207110438-010-005.jpg'
RESIZE = 384
CLASS_NAMES = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

class Inferencer:
    def __init__(self, tflite_path, resize=384):
        self.interpreter = load_tflite_model(tflite_path)
        self.resize = (resize, resize)
        self.face_landmarker = MediaPipe()

    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0) # batch 차원 추가 : (H, W, C) -> (1, H, W, C)
        return img
    
    def predict(self, input_np, sector):
        # 모델 설계할 때 sector_input의 shape은 (1,)로 만들었지만 tflite로 변환할 때는 (-1, 1)로 변환해야함
        sector_id = np.array(SECTOR2IDX[sector], dtype=np.int8).reshape((-1, 1))

        # tflite가 받는 입출력 tensor 정보 가져오기
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # 원본의 inputs는 [image_input, sector_input] 순이지만 tflite로 변환할 때에는 LIFO로 바뀜
        self.interpreter.set_tensor(input_details[0]['index'], sector_id)
        self.interpreter.set_tensor(input_details[1]['index'], input_np)
        self.interpreter.invoke()

        # output은 하나만 있으므로 그대로 갖다 사용
        output_index = output_details[0]['index']

        output_data = self.interpreter.get_tensor(output_index)
        probs = output_data[0]

        pred_class_idx = np.argmax(probs)

        return pred_class_idx, probs

    def run(self, image_path):
        with open(image_path, 'rb') as f:
            image = f.read()

        mp_image = self.face_landmarker.create_mp_image(image)
        detection_result = self.face_landmarker.landmarker.detect(mp_image)
        landmarks = detection_result.face_landmarks[0]

        np_image = mp_image.numpy_view()
        masked_image = remove_non_skin(np_image, landmarks)

        parts = dict(zip(SECTORS, wrinkle_preprocess(masked_image, landmarks)))
        
        for key, value in parts.items():
            # 이미지 제대로 처리되고 있는지 확인용. cv2.imwrite()는 BGR로 받길 예상하고 RGB로 저장하기 때문에 혼란스러움
            # Image.fromarray(value).save(f"/home/work/hocheol_dir/workspace/tmp/wrinkle_tmp/{key}_raw.jpg")
            parts[key] = self.preprocess(value)
            # Image.fromarray(parts[key][0].astype(np.uint8)).save(f"/home/work/hocheol_dir/workspace/tmp/wrinkle_tmp/{key}_process.jpg")
        
        preds, probs = {}, {}
        for sector in SECTORS:
            preds[sector], probs[sector] = self.predict(parts[sector], sector)
        
        return preds, probs

if __name__ == '__main__':
    inferencer = Inferencer(TFLITE_PATH, resize=RESIZE)
    preds, probs = inferencer.run(IMAGE_PATH)

    col_names = ['image_path', 'sector', 'predicted_class'] + CLASS_NAMES
    rows = []
    for sector in SECTORS:
        rows.append([IMAGE_PATH, sector, preds[sector]] + list(probs[sector]))
    df = pd.DataFrame(rows, columns=col_names)
    print('wrinkle tflite 결과')
    print(df)

end = datetime.now()
print(f"소요 시간: {end - start}")