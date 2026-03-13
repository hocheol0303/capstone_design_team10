# pip: tf2onnx, apt: onnxruntime가 설치되어 있어야 한다.
import os
import tensorflow as tf
import shutil
import subprocess

# seaborn 안쓰니까 있는척 속이고 utils 함수 import
import types
import sys
sys.modules['seaborn'] = types.ModuleType('seaborn')

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
    pig_3ch
    * pig_6ch
    
    * age_3section
    * age_1section
    age_cls
    wrinkle
'''

TASK = 'pig_3ch'
INPUT_SIZE = 384

WANDB_MODEL = f'hobbanglab/{TASK}/250821_162100_S-model:epoch10'

if TASK == 'pig_6ch':
    pass
elif TASK == 'age_3section':
    pass
elif TASK == 'age_1section':
    pass
else:
    # age_cls, pig_3ch, wrinkle
    SHAPE = (INPUT_SIZE, INPUT_SIZE, 3)

def keras2trt(model_path, onnx_dir, trt_dir, task, shape, tmp_dir, epoch):
    model = load_wandb_model(model_path)
    model.trainable = False

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    if task == 'pig_6ch':
        pass
    elif task == 'age_3section':
        pass
    elif task == 'age_1section':
        pass
    elif task == 'wrinkle':
        dummy_input = [tf.random.uniform((1, *shape)), tf.random.uniform((1,), dtype=tf.int32, minval=0, maxval=7)]
    else:
        # age_cls, pig_3ch, wrinkle
        dummy_input = tf.random.uniform((1, *shape))

    _ = model(dummy_input)
    model.export(tmp_dir)
    
    onnx_file = os.path.join(onnx_dir, f"{epoch}.onnx")
    cmd = f"python3 -m tf2onnx.convert --saved-model {tmp_dir} --output {onnx_file} --opset 13"
    subprocess.run(cmd, shell=True, check=True)
    print(f"ONNX 변환 완료: {onnx_file}")

    trt_file = os.path.join(trt_dir, f"{epoch}.trt")
    cmd = f"trtexec --onnx={onnx_file} --saveEngine={trt_file}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"TensorRT 변환 완료: {trt_file}")

    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    run_name = re.search(rf"{TASK}/.+-model", WANDB_MODEL).group(0).replace('-model', '').replace(f"{TASK}/", '')
    epoch = re.search(r"epoch[0-9]+", WANDB_MODEL).group(0)
    
    onnx_dir = f"/home/work/hocheol_dir/workspace/inference/TensorRT/onnx/{TASK}/{run_name}"
    trt_dir = f"/home/work/hocheol_dir/workspace/inference/TensorRT/TRT/{TASK}/{run_name}"

    # .keras -> .trt 변환할 때 임시 저장할 디렉터리
    tmp_dir = "/home/work/hocheol_dir/workspace/tmp_saved_model"

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(trt_dir, exist_ok=True)

    keras2trt(WANDB_MODEL, onnx_dir, trt_dir, TASK, SHAPE, tmp_dir, epoch)