import os
import tensorflow as tf

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import pandas as pd
import dataloaders
from tqdm import tqdm
from utils.constants import SECTORS

import re
import yaml
from datetime import datetime

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
# 1107 기존: /home/work/hocheol_dir/workspace/save_models/wrinkle/v0.2/250819_002635_S/epoch100_250819_002635.keras
# 1107 새: /home/work/hocheol_dir/workspace/save_models/wrinkle/v0.3/251014_142632_S/epoch70_251014_142632.keras
# MODEL_PATH = '/home/work/hocheol_dir/workspace/save_models/wrinkle/v0.3/251014_142632_S/epoch70_251014_142632.keras'
CONFIG_PATH = '/home/hocheol/inskin_ai/EfficientNet/src/configs/wrinkle_test.yaml'

COL_NAMES = ['path', 'sector', 'true', 'pred'] + [f"conf_{i}" for i in range(5)]

# tf.function으로 감싸서 그래프 재생성 방지
@tf.function
def predict_step(model, inputs):
    return model(inputs, training=False)

def predict(model, dataset):
    results = []

    for batch in tqdm(dataset):
        inputs, labels, paths = batch
        sector_ids = inputs['sector_input']

        wrinkle_preds = predict_step(model, inputs)
        pred_classes = tf.argmax(wrinkle_preds, axis=1)

        preds_np = wrinkle_preds.numpy()
        pred_classes_np = pred_classes.numpy()
        sector_ids_np = sector_ids.numpy()
        labels_np = labels.numpy()
        paths_np = [p.decode('utf-8') for p in paths.numpy()]
        
        for i in range(len(paths_np)):
            omitted_path = re.search(r"/[0-9]+_data/.+", paths_np[i]).group(0).lstrip('/')
            results.append([
                omitted_path,
                sector_ids_np[i],
                labels_np[i],
                pred_classes_np[i]] + preds_np[i].tolist()
            )

    return results

if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    dataset = dataloaders.wrinkle.get_test_datasets(config=config)
    model = tf.keras.models.load_model(config['model_path'], compile=False)
    results = predict(model, dataset)

    df = pd.DataFrame(results, columns=COL_NAMES)
    
    df.sort_values(by='path', inplace=True)

    result_path = os.path.join(
        config['result_dir'],
        *config['model_path'].split('/')[-4:-1],
        f"{os.path.splitext(config['model_path'].split('/')[-1])[0]}_{datetime.now().strftime('%y%m%d_%H%M%S')}-{config['comment']}.csv"
    )

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")