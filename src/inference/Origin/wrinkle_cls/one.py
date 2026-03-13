import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from utils.Mediapipe import MediaPipe
from datetime import datetime
import pandas as pd
from utils.constants import SECTORS, SECTOR2IDX, load_wandb_model
import sys
sys.path.append('/home/work/hocheol_dir/workspace/_dataset/preprocess/wrinkle/v0.3/v0.3.0')
from wrinkle_preprocess import wrinkle_preprocess
import re

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

'''
모델 저장할 당시의 input size로 지정해야함

EfficientNetV2-B0 : (224, 224)
EfficientNetV2-B1 : (240, 240)
EfficientNetV2-B2 : (260, 260)
EfficientNetV2-B3 : (300, 300)
EfficientNetV2-S : (384, 384)
EfficientNetV2-M : (480, 480)
EfficientNetV2-L : (480, 480)
EfficientNetV2-XL (비공식) : (512, 512)
'''

# MODEL_PATH = "/home/work/hocheol_dir/workspace/save_models/wrinkle/v0.3/251014_142632_S/epoch70_251014_142632.keras"
WANDB_MODEL = 'hobbanglab/wrinkle/251014_142632_S-model:v6'

RESIZE = 384
IMAGE_PATH = "/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/030_data/data/1207_face.jpg"
CLASS_NAMES = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']

class Inferencer:
    def __init__(self, model_path, resize=(384, 384)):
        # self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model = load_wandb_model(model_path)
        self.resize = resize
        self.face_landmarker = MediaPipe()

    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, input_np, sector):
        sector_id = tf.constant(SECTOR2IDX[sector], dtype=tf.int32, shape=(1,))
        input_tensor = tf.constant(input_np, dtype=tf.float32)
        inputs = {'image_input': input_tensor, 'sector_input': sector_id}
        preds = self.model(inputs, training=False)
        probs = preds[0].numpy()
        pred_class = np.argmax(probs)

        return pred_class, probs
        

    def run(self, image_path):
        with open(image_path, 'rb') as f:
            image = f.read()
        
        mp_image = self.face_landmarker.create_mp_image(image)
        detection_result = self.face_landmarker.landmarker.detect(mp_image)
        landmarks = detection_result.face_landmarks[0]

        np_image = mp_image.numpy_view()
        parts = dict(zip(SECTORS, wrinkle_preprocess(np_image, landmarks)))

        for key, value in parts.items():
            parts[key] = self.preprocess(value)
        
        preds, probs = {}, {}
        for sector in SECTORS:
            preds[sector], probs[sector] = self.predict(parts[sector], sector)
        
        return preds, probs

if __name__ == '__main__':
    # 이제 모델 저장 경로가 wandb와 의존성이 생김
    run_name = re.search(r"/.+-model", WANDB_MODEL).group(0).replace('-model', '')
    # epoch = re.search(r"epoch[0-9]+", WANDB_MODEL).group(0)
    # output_path = f"/home/work/hocheol_dir/workspace/inference/results/v0.2/wrinkle/{run_name}/{epoch}_origin.csv"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    inferencer = Inferencer(WANDB_MODEL, resize=(RESIZE, RESIZE))
    preds, probs = inferencer.run(IMAGE_PATH)

    col_names = ['image_path', 'sector', 'predicted_class'] + CLASS_NAMES
    rows = []
    for sector in SECTORS:
        rows.append([IMAGE_PATH, sector, preds[sector]] + probs[sector].tolist())

    df = pd.DataFrame(rows, columns=col_names)
    print('wrinkle origin 결과')
    print(df)
    # df.to_csv(output_path, index=False)