import os
import tensorflow as tf
# GPU를 사용할만큼만 메모리 할당
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import pandas as pd
from dataloaders.pig_3ch import get_test_dataset
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
# 1107 기존: /home/work/hocheol_dir/workspace/save_models/pig_3ch/v0.3/250909_211855_S/epoch70_250909_211855.keras
# 1107 새: /home/work/hocheol_dir/workspace/save_models/pig_3ch/v0.4/251015_084023_S/epoch70_251015_084023.keras
# MODEL_PATH = "/home/hocheol/inskin_ai/no_track/save_models/pig_3ch/v0.4/251015_084023_S/epoch70_251015_084023.keras" 
CONFIG_PATH = '/home/hocheol/inskin_ai/EfficientNet/src/configs/pig_test.yaml'

RESULTS_DIR = '/home/hocheol/inskin_ai/no_track/inference_results/effnet'
COL_NAMES = ['path', 'label', 'pred', 'conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4']

@tf.function
def predict_step(model, images):
    preds = model(images, training=False)
    return preds

def predict(model, dataset):
    results = []

    for batch in tqdm(dataset):
        images, labels, paths = batch
        preds = predict_step(model, images)
        probs = tf.nn.softmax(preds, axis=-1)
        pred_labels = tf.argmax(probs, axis=-1)

        for path, label, pred_label, prob in zip(paths, labels, pred_labels, probs):
            path_str = path.numpy().decode('utf-8')
            omitted_path = re.search(r"/[0-9]+_data/.+", path_str).group(0).lstrip('/')
            results.append([omitted_path, label.numpy(), pred_label.numpy()] + prob.numpy().tolist())

    return results
            
if __name__ == '__main__':
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 로드
    dataset = get_test_dataset(config=config)

    # 모델 로드
    model = tf.keras.models.load_model(config['model_path'], compile=False)

    # 예측 수행
    results = predict(model, dataset)

    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results, columns=COL_NAMES)
    df[COL_NAMES[3:]] = df[COL_NAMES[3:]].round(3)
    df.sort_values(by='path', inplace=True)

    # 결과 저장
    result_path = os.path.join(
        RESULTS_DIR,
        *config['model_path'].split('/')[-4:-1],
        f"{os.path.splitext(config['model_path'].split('/')[-1])[0]}_{datetime.now().strftime('%y%m%d_%H%M%S')}_batch.csv"
    )

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")