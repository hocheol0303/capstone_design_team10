import os
import tensorflow as tf
# GPU를 사용할만큼만 메모리 할당
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import pandas as pd
from _dataloaders.age_cls import get_test_datasets
from tqdm import tqdm
import re
import yaml
from datetime import datetime
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

RESIZE = (384, 384)
MODEL_PATH = "/home/work/hocheol_dir/workspace/save_models/age_cls/v0.3/250921_122419_S/epoch70_250921_122419.keras"
CONFIG_PATH = '/home/work/hocheol_dir/workspace/configs/age_test.yaml'

RESULTS_DIR = '/home/work/hocheol_dir/workspace/inference/results'
COL_NAMES = ['path', 'gender', 'age', 'label', 'pred', 'conf_10', 'conf_20', 'conf_30', 'conf_40', 'conf_50', 'conf_60', 'conf_70']

def predict(model, dataset):
    results = []

    for batch in tqdm(dataset):
        images, labels, weights, paths, ages = batch
        preds = model(images, training=False)

        for i in range(images.shape[0]):
            path_str = paths[i].numpy().decode('utf-8')
            omitted_path = re.search(r"/[0-9]+_data/.+", path_str).group(0).lstrip('/')

            if weights['male'][i].numpy() == 1.0:
                # labes는 sparse tensor이고 preds one-hot tensor이므로 둘의 형태가 다른 것
                true = labels['male'][i].numpy()
                pred = tf.argmax(preds[0][i]).numpy()   # age_cls의 model 보면 ['male', 'female'] 순서대로 출력
                probs = preds[0][i].numpy()
                gender = 'male'
            else:
                true = labels['female'][i].numpy()
                pred = tf.argmax(preds[1][i]).numpy()
                probs = preds[1][i].numpy()
                gender = 'female'
            
            age = ages[i].numpy()
            
            results.append([omitted_path, gender, age, true, pred] + probs.tolist())
    
    return results
            
if __name__ == '__main__':
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 데이터셋 로드
    dataset = get_test_datasets(config=config)

    # 모델 로드
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 예측 수행
    results = predict(model, dataset)

    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results, columns=COL_NAMES)
    df[COL_NAMES[3:]] = df[COL_NAMES[3:]].round(3)
    df.sort_values(by='path', inplace=True)

    # 결과 저장
    result_path = os.path.join(
        RESULTS_DIR,
        *MODEL_PATH.split('/')[-4:-1],
        f"{os.path.splitext(MODEL_PATH.split('/')[-1])[0]}_{datetime.now().strftime('%y%m%d_%H%M%S')}_batch.csv"
    )

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    df.to_csv(result_path, index=False)