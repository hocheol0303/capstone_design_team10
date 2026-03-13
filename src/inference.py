import json
import os
import csv
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from dataset_loader import get_test_datasets
import random
import re
from utils.slack import send_slack_message

# GPU를 사용할만큼만 메모리 할당
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

'''
TASK:
    pigmentation
    # age
    age_only1
'''

TASK = 'age_only1'
CHANNEL = 3
BATCH_SIZE = 64

model_root = '/home/work/hocheol_dir/workspace/tmp/save_models/age_only1_S_3ch_384_100Epoch_160Batch/20250522_171356'
# breakpoint()

MODELS = []
for file_name in os.listdir(model_root):
    if file_name.endswith('.keras'):
        if int(re.search(r'epoch(\d+)', file_name).group(1)) % 10 == 0:
            MODELS.append(os.path.join(model_root, file_name))

JSON_PATH = '/home/work/hocheol_dir/workspace/data/data_in_use/0522_split_5Fold/fold_0/test_fold_0.json'
RESULTS_DIR = os.path.join('/home/work/hocheol_dir/workspace/tmp/inference_results', f'{TASK}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
IMG_SIZE = (384, 384)

ROOT_DIR = '/home/work/hocheol_dir/workspace/data/data_in_use'

os.makedirs(os.path.join(RESULTS_DIR), exist_ok=True)

test_dataset = get_test_datasets(
    JSON_PATH,
    root_dir=ROOT_DIR,
    batch_size=BATCH_SIZE,
    channel=CHANNEL,
    target_size=IMG_SIZE,
    task=TASK,
)
try:
    for i, model_path in enumerate(MODELS):
        # re를 사용하여 epoch 추출
        epoch = re.search(r'epoch(\d+)', model_path).group(1)

        if not model_path.endswith('.keras'):
            continue
        # Keras 모델 로드
        model = tf.keras.models.load_model(model_path, compile=False)

        results = []
        if TASK == 'pigmentation':
            col_names = ['path', 'label', 'pred', 'conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4']
        elif TASK == 'age':
            col_names = ['forehead_path', 'left_cheek_path', 'right_cheek_path', 'true_age', 'pred_forehead', 'pred_left_cheek', 'pred_right_cheek', 'pred_average']
        elif TASK == 'age_only1':
            col_names = ['path', 'true_age', 'pred_age']

        # 추론 수행
        if TASK == 'pigmentation':
            y_true, y_pred, all_probs, left_paths, right_paths = [], [], [], [], []
            for batch in tqdm(test_dataset):
                images, labels, (left_path, right_path) = batch
                preds = model(images, training=False)
                probs = tf.nn.softmax(preds, axis=-1)
                pred_labels = tf.argmax(probs, axis=-1)

                all_probs.extend(probs.numpy())
                y_true.extend(labels.numpy().tolist())
                y_pred.extend(pred_labels.numpy().tolist())
                left_paths.extend(left_path.numpy())
                right_paths.extend(right_path.numpy())

            for left_path, right_path, true_label, pred_label, prob in zip(left_paths, right_paths, y_true, y_pred, all_probs):
                if isinstance(left_path, bytes):
                    left_path = left_path.decode()
                if isinstance(right_path, bytes):
                    right_path = right_path.decode()
                paths_str = f"{left_path} | {right_path}"
                results.append([paths_str, true_label, pred_label] + prob.tolist())
                

        # age는 당장 안쓸 것 같아서 작전상 후퇴. 쓰려면 dataset_loader.py에서 수정할 것
        # elif TASK == 'age':
        #     y_true, y_pred, path_list = [], [], []
        #     for batch in tqdm(test_dataset):
        #         images, labels, paths = batch


        #     for item in tqdm(test_meta):
        #         f_path = os.path.join(ROOT_DIR, item['forehead_path'])
        #         l_path = os.path.join(ROOT_DIR, item['left_cheek_path'])
        #         r_path = os.path.join(ROOT_DIR, item['right_cheek_path'])
        #         age = float(item['age'])

        #         f_img = load_img(f_path, IMG_SIZE)
        #         l_img = load_img(l_path, IMG_SIZE)
        #         r_img = load_img(r_path, IMG_SIZE)
        #         input_tensor = [tf.expand_dims(f_img, axis=0),
        #                         tf.expand_dims(l_img, axis=0),
        #                         tf.expand_dims(r_img, axis=0)]
                
        #         preds = model(input_tensor, training=False)
        #         pred_forehead = float(preds[0][0])
        #         pred_left_cheek = float(preds[1][0])
        #         pred_right_cheek = float(preds[2][0])
        #         pred_average = float(preds[3][0])

        #         results.append([
        #             f_path,
        #             l_path,
        #             r_path,
        #             age,
        #             round(pred_forehead, 2),
        #             round(pred_left_cheek, 2),
        #             round(pred_right_cheek, 2),
        #             round(pred_average, 2)
        #         ])
            
        elif TASK == 'age_only1':
            y_true, y_pred, all_paths = [], [], []
            for batch in tqdm(test_dataset):
                images, ages, paths = batch
                pred = model(images, training=False)

                y_true.extend(ages.numpy().tolist())
                y_pred.extend(pred.numpy().tolist())
                all_paths.extend(paths.numpy().tolist())

            # breakpoint()
            for path, true_age, pred_age in zip(all_paths, y_true, y_pred):
                if isinstance(path, bytes):
                    path = path.decode()
                results.append([path, true_age, pred_age[0]])

        # CSV 저장
        with open(os.path.join(RESULTS_DIR, f"inference_epoch{epoch}.csv"), mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            writer.writerows(results)

        with open(os.path.join(RESULTS_DIR, f'info_epoch{epoch}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Task: {TASK}\n")
            f.write(f"Image Size: {IMG_SIZE}\n")
            f.write(f"Channel: {CHANNEL}\n")
            f.write(f"JSON Path: {JSON_PATH}\n")

        print(f"✅ CSV 저장 완료: {RESULTS_DIR}")

        # 메모리 해제
        del model
        tf.keras.backend.clear_session()
        
    send_slack_message('Inference 완료!')

except Exception as e:
    send_slack_message(f'Inference 중 오류 발생: {str(e)}')
    raise e