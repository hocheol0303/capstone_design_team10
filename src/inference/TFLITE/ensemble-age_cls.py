from datetime import datetime
start = datetime.now()

import os
import cv2
import csv
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
from _dataset.preprocess.face_crop_roll import age_preprocess
from utils.Mediapipe import MediaPipe
import re

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

GENDER:
    male
    female
'''
# 새 학습 모델
TFLITE_PATH1 = '/home/work/hocheol_dir/workspace/inference/results/age_cls/v0.3/251013_110039_S/tflite/epoch70.tflite'
# 0808 재현 모델
TFLITE_PATH2 = '/home/work/hocheol_dir/workspace/inference/results/age_cls/v0.3.1/250924_091410_S/epoch70_250924_091410/tflite/epoch70.tflite'

IMAGE_PATH = '/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/034_data/data/0101_CRS_19_01.jpg'
GENDER = 'female'
RESIZE = 384
CLASS_NAMES = ['10', '20', '30', '40', '50', '60', '70']

RESULTS_DIR = '/home/work/hocheol_dir/workspace/inference/results'

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

class Inferencer:
    def __init__(self, tflite_path1, tflite_path2, resize=(384, 384)):
        self.interpreter1 = load_tflite_model(tflite_path1)
        self.interpreter2 = load_tflite_model(tflite_path2)
        self.resize = resize
        self.face_landmarker = MediaPipe()
        
    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # 배치차원 추가 : (H, W, C) -> (1, H, W, C)
        return img


    def predict(self, input_np, gender):
        # 모델의 입출력 정보는 동일하다고 가정하고 하나만 사용
        input_details = self.interpreter1.get_input_details()
        output_details = self.interpreter1.get_output_details()

        # interpreter의 입력 인덱스에 이미지 배열을 넣기
        # 이 부분에서 입력 이미지는 전처리가 완료된 상황이어야 한다.
        self.interpreter1.set_tensor(input_details[0]['index'], input_np)

        # interpreter.invoke()는 모델의 forward pass를 실행하는 함수
        self.interpreter1.invoke()
        
        # models/age_cls.py를 보면 model의 output이 male, female 순서대로이므로 0번에 male을, 1번에 female을 할당한다.
        # 멀티 태스크라서 헤드별로 다른 output이 있음
        for detail in output_details:
            if gender == 'male' and detail['name'].endswith(':0'):
                output_index = detail['index']
            elif gender == 'female' and detail['name'].endswith(':1'):
                output_index = detail['index']

        output1 = self.interpreter1.get_tensor(output_index)

        self.interpreter2.set_tensor(input_details[0]['index'], input_np)
        self.interpreter2.invoke()
        for detail in output_details:
            if gender == 'male' and detail['name'].endswith(':0'):
                output_index = detail['index']
            elif gender == 'female' and detail['name'].endswith(':1'):
                output_index = detail['index']
        
        output2 = self.interpreter2.get_tensor(output_index)
        probs = ((output1+output2)/2.0)[0]
        pred_class = np.argmax(probs)

        return pred_class, probs

    def run(self, image_path, gender):
        with open(image_path, 'rb') as f:
            image = f.read()
        
        mp_image = self.face_landmarker.create_mp_image(image)
        detection_result = self.face_landmarker.landmarker.detect(mp_image)
        landmarks = detection_result.face_landmarks[0]
        
        preprocessed_image = age_preprocess(landmarks, mp_image)
        preprocessed_image = self.preprocess(preprocessed_image)

        pred, probs = self.predict(preprocessed_image, gender)

        return pred, probs


if __name__ == "__main__":
    inferencer = Inferencer(TFLITE_PATH1, TFLITE_PATH2, resize=(RESIZE, RESIZE))
    pred_class, probs = inferencer.run(IMAGE_PATH, GENDER)

    relative_image_path = re.search(r"datasets/.+", IMAGE_PATH).group(0)

    col_names = ['image_path', 'predicted_class'] + [f"conf_{i*10}" for i in range(1, len(CLASS_NAMES)+1)]
    outputs = [relative_image_path, CLASS_NAMES[pred_class]] + list(probs)

    print(f"예측 클래스: {CLASS_NAMES[pred_class]}")
    print(f"예측 확률: {probs}")

    df = pd.DataFrame([outputs], columns=col_names)
    print(df)

    now = datetime.now().strftime('%y%m%d_%H%M%S')
    # 결과 저장
    result_path = os.path.join(
        RESULTS_DIR,
        'age_cls',
        'ensemble',
        now,
        'results.csv'
    )
    # os.makedirs(os.path.dirname(result_path), exist_ok=True)
    # df.to_csv(result_path, index=False)

    # with open(os.path.join(os.path.dirname(result_path), 'model_info.txt'), 'w') as f:
    #     f.write(f"Model 1: {TFLITE_PATH1}\n")
    #     f.write(f"Model 2: {TFLITE_PATH2}\n")

end = datetime.now()
print(f"소요 시간: {end - start}")