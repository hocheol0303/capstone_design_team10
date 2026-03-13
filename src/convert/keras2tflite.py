import os
import gc
import keras.backend as K
import tensorflow as tf
import shutil
from utils.wandb import load_wandb_model
import re

# GPU를 사용할만큼만 메모리 할당
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

'''
pigmentation model input shape = (384, 384, 6) 형태
age model input size = [(384, 384, 3)]*3 형태:
    forehead_input, left_cheek_input, right_cheek_input을 받기 때문

TASK:
    # pigmentation
    # age
    # age_only1

    age_cls
    pig_3ch
    wrinkle
'''

VERSION = 'v0.3'
TASK = 'wrinkle'
INPUT_SIZE = 384

# WANDB_MODEL = f'hobbanglab/{TASK}/250821_162100_S-model:epoch10'
MODEL_PATH = "/home/work/hocheol_dir/workspace/save_models/wrinkle/v0.3/251014_142632_S/epoch70_251014_142632.keras"
RESULTS_DIR = '/home/work/hocheol_dir/workspace/inference/results'

if TASK == 'pig_6ch':
    SHAPE = (INPUT_SIZE, INPUT_SIZE, 6)
elif TASK == 'age':
    SHAPE = [(INPUT_SIZE, INPUT_SIZE, 3)] * 3
else:
    SHAPE = (INPUT_SIZE, INPUT_SIZE, 3)


# .keras -> .tflite 변환할 때 임시 저장할 디렉터리
TMP_DIR = "/home/work/hocheol_dir/workspace/tmp_saved_model"

os.makedirs(TMP_DIR, exist_ok=True)

def keras2tflite(model, tflite_path, task, shape):
    if task == 'age':
        dummy_input = [tf.random.uniform((1, *s) for s in shape)]    
    elif task == 'wrinkle':
        dummy_input = [tf.random.uniform((1, *shape)), tf.random.uniform((1,), dtype=tf.int32, minval=0, maxval=7)]
    else:
        dummy_input = tf.random.uniform((1, *shape))

    _ = model(dummy_input)
    model.export(TMP_DIR)

    converter = tf.lite.TFLiteConverter.from_saved_model(TMP_DIR)
    converter.optimizations = []
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ 변환 완료: {tflite_path}")

    shutil.rmtree(TMP_DIR)
    print("✅ tmp 디렉터리 삭제 완료")

if __name__ == "__main__":
    run_name = re.search(r"[0-9]+_[0-9]+_[A-Z]*", MODEL_PATH).group(0) # 날짜_크기
    epoch = re.search(r"epoch[0-9]+", MODEL_PATH).group(0) # epoch##

    # 로컬
    task = MODEL_PATH.split('/')[-4]
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # wandb
    # task = MODEL_PATH.split('/')[-2]
    # model = load_wandb_model(MODEL_PATH)

    model.trainable = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    result_path = os.path.join(
        RESULTS_DIR,
        task,
        VERSION,
        run_name,
        'tflite',
        f"{epoch}.tflite"
    )

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    keras2tflite(model, result_path, TASK, SHAPE)