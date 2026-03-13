import os
import tensorflow as tf
# GPU를 사용할만큼만 메모리 할당
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import pandas as pd
from dataloaders.age_cls import get_test_datasets
from tqdm import tqdm
import re
from datetime import datetime
import yaml

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
MODEL_PATH1 = "/home/hocheol/inskin_ai/no_track/save_models/age_cls/v0.3.1/250924_091410_S/epoch70_250924_091410.keras"
MODEL_PATH2 = "/home/hocheol/inskin_ai/no_track/save_models/age_cls/v0.3/251013_110039_S/epoch70_251013_110039.keras"

CONFIG_PATH = '/home/hocheol/inskin_ai/EfficientNet/src/configs/age_test.yaml'

RESULTS_DIR = '/home/hocheol/inskin_ai/no_track/inference_results/effnet'
COL_NAMES = ['path', 'gender', 'age', 'label', 'pred', 'conf_10', 'conf_20', 'conf_30', 'conf_40', 'conf_50', 'conf_60', 'conf_70']

@tf.function
def predict_step(model1, model2, images):
    preds1 = model1(images, training=False)
    preds2 = model2(images, training=False)
    return preds1, preds2

def predict(model1, model2, dataset):
    results = []

    for batch in tqdm(dataset):
        images, labels, weights, paths, ages = batch
        preds1, preds2 = predict_step(model1, model2, images)

        for i in range(images.shape[0]):
            path_str = paths[i].numpy().decode('utf-8')
            omitted_path = re.search(r"/[0-9]+_data/.+", path_str).group(0).lstrip('/')

            if weights['male'][i].numpy() == 1.0:
                # labes는 sparse tensor이고 preds one-hot tensor이므로 둘의 형태가 다른 것
                true = labels['male'][i].numpy()
                
                pred1 = tf.argmax(preds1[0][i]).numpy()   # age_cls의 model 보면 ['male', 'female'] 순서대로 출력
                probs1 = preds1[0][i].numpy()

                pred2 = tf.argmax(preds2[0][i]).numpy()
                probs2 = preds2[0][i].numpy()

                gender = 'male'
            else:
                true = labels['female'][i].numpy()
                pred1 = tf.argmax(preds1[1][i]).numpy()
                probs1 = preds1[1][i].numpy()

                pred2 = tf.argmax(preds2[1][i]).numpy()
                probs2 = preds2[1][i].numpy()
                gender = 'female'
            
            age = ages[i].numpy()

            avg_probs = (probs1 + probs2) / 2.0
            # breakpoint()
            pred = avg_probs.argmax()
            
            results.append([omitted_path, gender, age, true, pred] + avg_probs.tolist())
    
    return results
if __name__ == '__main__':
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 데이터셋 로드
    dataset = get_test_datasets(config)

    # 모델 로드
    model1 = tf.keras.models.load_model(MODEL_PATH1, compile=False)
    model2 = tf.keras.models.load_model(MODEL_PATH2, compile=False)

    # 예측 수행
    results = predict(model1, model2, dataset)

    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results, columns=COL_NAMES)
    df[COL_NAMES[3:]] = df[COL_NAMES[3:]].round(3)
    df.sort_values(by='path', inplace=True)

    now = datetime.now().strftime('%y%m%d_%H%M%S')
    # 결과 저장
    result_path = os.path.join(
        RESULTS_DIR,
        'age_cls',
        'ensemble',
        now,
        'result_batch.csv'
    )

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(os.path.join(os.path.dirname(result_path), 'model_info.txt'), 'w') as f:
        f.write(f"Model 1: {MODEL_PATH1}\n")
        f.write(f"Model 2: {MODEL_PATH2}\n")
        f.write(f"Input Size: {RESIZE}\n")
        f.write(f"Batch Size: {config['batch_size']}\n")

    df.to_csv(result_path, index=False)