import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from _dataset.preprocess.face_crop_roll import age_preprocess
from utils.Mediapipe import MediaPipe
from datetime import datetime
import pandas as pd

# GPU를 사용할만큼만 메모리 할당
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

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

RESIZE = 384
MODEL_PATH = "/home/work/hocheol_dir/workspace/save_models/age_cls/v0.3.1/250924_091410_S/epoch70_250924_091410.keras"
IMAGE_PATH = "/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/034_data/data/0101_CRS_19_01.jpg"
GENDER = 'female'
CLASS_NAMES = ['10', '20', '30', '40', '50', '60', '70']

class Inferencer:
    def __init__(self, model_path, resize=(384, 384)):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.resize = resize
        self.face_landmarker = MediaPipe()
    
    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # 배치차원 추가 : (H, W, C) -> (1, H, W, C)
        return img
    
    def predict(self, input_img, gender):
        preds = self.model.predict(input_img)
        if gender == 'male':
            probs = preds[0]
        elif gender == 'female':
            probs = preds[1]
        else:
            raise ValueError(f"성별: {gender}는 지원하지 않습니다.")
        
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

        pred_class, probs = self.predict(preprocessed_image, gender)

        return pred_class, probs

if __name__ == '__main__':
    inferencer = Inferencer(MODEL_PATH, (RESIZE, RESIZE))
    pred_class, probs = inferencer.run(IMAGE_PATH, GENDER)

    col_names = ['image_path', 'predicted_class'] + [f"conf_{label}" for label in CLASS_NAMES]
    outputs = [IMAGE_PATH, CLASS_NAMES[pred_class]] + probs.tolist()[0]

    df = pd.DataFrame([outputs], columns=col_names)
    print(df)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # df.to_csv(output_path, index=False)