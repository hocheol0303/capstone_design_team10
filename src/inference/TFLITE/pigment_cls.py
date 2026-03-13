from datetime import datetime
start = datetime.now()

import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import mediapipe as mp

from _dataset.preprocess.remove_non_skin import remove_non_skin
from _dataset.preprocess.pigment_data.crop_mp import crop_left_cheek_from_np_image, crop_right_cheek_from_np_image
from utils.Mediapipe import MediaPipe

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
'''

TFLITE_PATH = '/home/work/hocheol_dir/workspace/inference/results/pig_3ch/v0.4/251015_084023_S/tflite/epoch70.tflite'
IMAGE_PATH = '/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/045_data/data/F0085_IND_GM_68_0_01.JPG'
RESIZE = 384
CLASS_NAMES = ['0', '1', '2', '3', '4']

RESULTS_DIR = '/home/work/hocheol_dir/workspace/inference/results'

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def load_tflite_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

class Inferencer:
    def __init__(self, tflite_path, resize=(384, 384)):
        self.interpreter = load_tflite_model(tflite_path)
        self.resize = resize
        self.face_landmarker = MediaPipe()
        
    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # 배치차원 추가 : (H, W, C) -> (1, H, W, C)
        return img

    def predict(self, image):
        # tflite가 받는 입출력 tensor 정보 가져오기
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # interpreter의 입력 인덱스에 이미지 배열을 넣기
        # 이 부분에서 입력 이미지는 전처리가 완료된 상황이어야 한다.
        self.interpreter.set_tensor(input_details[0]['index'], image)

        # interpreter.invoke()는 모델의 forward pass를 실행하는 함수
        self.interpreter.invoke()
        
        output_index = output_details[0]['index']

        output_data = self.interpreter.get_tensor(output_index)
        probs = softmax(output_data[0])

        pred_class = np.argmax(probs)

        return pred_class, probs
    
    def run(self, image_path):
        with open(image_path, 'rb') as f:
            image = f.read()
        
        mp_image = self.face_landmarker.create_mp_image(image)
        np_image = mp_image.numpy_view()
        detection_result = self.face_landmarker.landmarker.detect(mp_image)
        landmarks = detection_result.face_landmarks[0]

        masked_image = remove_non_skin(np_image, landmarks)

        left_cheek = crop_left_cheek_from_np_image(masked_image, landmarks)
        right_cheek = crop_right_cheek_from_np_image(masked_image, landmarks)

        preprocessed_left = self.preprocess(left_cheek)
        preprocessed_right = self.preprocess(right_cheek)

        left_pred, left_probs = self.predict(preprocessed_left)
        right_pred, right_probs = self.predict(preprocessed_right)

        return (left_pred, left_probs), (right_pred, right_probs)

if __name__ == "__main__":
    inferencer = Inferencer(TFLITE_PATH, resize=(RESIZE, RESIZE))
    (left_pred, left_probs), (right_pred, right_probs) = inferencer.run(IMAGE_PATH)
    
    relative_image_path = re.search(r"datasets/.+", IMAGE_PATH).group(0)

    col_names = ['image_path', 'direction', 'preds'] + [f"conf_{i}" for i in CLASS_NAMES]
    outputs = []
    outputs.append([relative_image_path, 'left', CLASS_NAMES[left_pred]] + list(left_probs))
    outputs.append([relative_image_path, 'right', CLASS_NAMES[right_pred]] + list(right_probs))

    df = pd.DataFrame(outputs, columns=col_names)
    print('pig_3ch tflite 결과')
    print(df)

    task = TFLITE_PATH.split('/')[-5]
    version = TFLITE_PATH.split('/')[-4]
    run_name = re.search(r"[0-9]+_[0-9]+_[A-Z]*", TFLITE_PATH).group(0) # 날짜_크기
    epoch = re.search(r"epoch[0-9]+", TFLITE_PATH).group(0) # epoch##

    result_path = os.path.join(
        RESULTS_DIR,
        task,
        version,
        run_name,
        f"{epoch}_tflite.csv"
    )
    # os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # df.to_csv(result_path, index=False)

end = datetime.now()
print(f"소요 시간: {end - start}")