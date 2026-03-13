from datetime import datetime
start = datetime.now()
import sys
sys.path.append('/home/work/hocheol_dir/workspace/_dataset/preprocess')
from remove_non_skin import remove_non_skin
import os
import cv2
import numpy as np
import tensorflow as tf

# GPU를 사용할만큼만 메모리 할당
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import pandas as pd
from _dataloaders.pig_3ch import get_test_datasets
from tqdm import tqdm
from _dataset.preprocess.crop_data import crop_cheeks_from_image, crop_cheeks_from_detection_result
from utils.Mediapipe import MediaPipe
from utils.wandb import load_wandb_model
import re
import mediapipe as mp


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

MODEL_PATH = "/home/work/hocheol_dir/workspace/save_models/pig_3ch/v0.4/251015_084023_S/epoch70_251015_084023.keras"
# MODEL_PATH = 'hobbanglab/pig_3ch/250821_162100_S-model:epoch10'
VERSION = 'v0.4'


RESIZE = 384
IMAGE_PATH = '/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/045_data/data/F0085_IND_GM_68_0_01.JPG'
CLASS_NAMES = ['0', '1', '2', '3', '4']

RESULTS_DIR = '/home/work/hocheol_dir/workspace/inference/results'

class Inferencer:
    def __init__(self, model, resize):
        self.model = model
        self.resize = resize if isinstance(resize, tuple) else (resize, resize)
        self.face_landmarker = MediaPipe()

    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, input_np):
        preds = self.model(input_np, training=False)
        probs = tf.nn.softmax(preds, axis=-1)
        pred_class = tf.argmax(probs, axis=-1)
        
        return pred_class.numpy()[0], probs.numpy()[0]
        
    
    def run(self, image_path):
        with open(image_path, 'rb') as f:
            image = f.read()
        
        mp_image = self.face_landmarker.create_mp_image(image)
        detection_result = self.face_landmarker.landmarker.detect(mp_image)
        landmarks = detection_result.face_landmarks[0]

        image_without_background = remove_non_skin(mp_image.numpy_view(), landmarks)
        mp_image_without_background = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_without_background)

        left_cheek, right_cheek = crop_cheeks_from_detection_result(landmarks, mp_image_without_background)
        left_cheek = self.preprocess(left_cheek)
        right_cheek = self.preprocess(right_cheek)

        left_pred_class, left_probs = self.predict(left_cheek)
        right_pred_class, right_probs = self.predict(right_cheek)

        return (left_pred_class, left_probs), (right_pred_class, right_probs)

# mp로 이미지 로드
if __name__ == '__main__':
    run_name = re.search(r"[0-9]+_[0-9]+_[A-Z]*", MODEL_PATH).group(0) # 날짜_크기
    epoch = re.search(r"epoch[0-9]+", MODEL_PATH).group(0) # epoch##

    # 로컬
    task = MODEL_PATH.split('/')[-4]
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    

    # wandb
    # task = MODEL_PATH.split('/')[-2] # wandb
    # model = load_wandb_model(MODEL_PATH)

    result_path = os.path.join(
        RESULTS_DIR,
        task,
        VERSION,
        run_name,
        f"{epoch}_origin.csv"
    )
    # os.makedirs(os.path.dirname(result_path), exist_ok=True)

    inferencer = Inferencer(model, resize=(RESIZE, RESIZE))
    (left_pred_class, left_probs), (right_pred_class, right_probs) = inferencer.run(IMAGE_PATH)

    relative_image_path = re.search(r"datasets/.+", IMAGE_PATH).group(0)

    col_names = ['image_path', 'direction', 'pred'] + [f"conf_{label}" for label in CLASS_NAMES]
    results = []
    results.append([relative_image_path, 'left', CLASS_NAMES[left_pred_class]]+list(left_probs))
    results.append([relative_image_path, 'right', CLASS_NAMES[right_pred_class]]+list(right_probs))
    
    df = pd.DataFrame(results, columns=col_names)
    print('pig_3ch origin 결과')
    print(df)
    # df.to_csv(result_path, index=False)

end = datetime.now()
print(f"소요 시간: {end - start}")