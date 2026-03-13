import os
import argparse
# # 모델 학습 시 그래프 찾는 데 시간이 오래 걸려서 뭔가 뭔가 고정하는 환경변수 설정
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
# os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
# os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import tensorflow as tf
# tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_jit(True)
tf.random.set_seed(42)


######################## 위에 설정은 학습 중간에 20기가 때문에 OOM이 떠서 건드림 ########################

# 1. XLA 관련 설정 최적화
# 'platform' 대신 기본 할당자를 사용하되, 사전 할당만 끕니다.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # <-- 이 줄은 주석 처리 추천

# 2. Autotune은 켜두는 것이 메모리/속도 면에서 유리합니다 (속도 고정이 꼭 필요할 때만 0 사용)
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2" 
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"

######################## 위에 설정은 학습 중간에 20기가 때문에 OOM이 떠서 건드림 ########################

# GPU를 사용할만큼만 메모리 할당
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import wandb
from utils.slack import send_slack_message
from datetime import datetime
import numpy as np
import random

import tasks
import dataloaders as dataloaders

import yaml

SEED = 42

# Python, NumPy, TensorFlow seed 고정
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 환경변수 설정
os.environ['PYTHONHASHSEED'] = str(SEED)

TEST_RUN = False

'''
EfficientNetV2-B0 : (224, 224)
EfficientNetV2-B1 : (240, 240)
EfficientNetV2-B2 : (260, 260)
EfficientNetV2-B3 : (300, 300)
EfficientNetV2-S : (384, 384)
EfficientNetV2-M : (480, 480)
EfficientNetV2-L : (480, 480)
EfficientNetV2-XL (비공식) : (512, 512)
'''

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), 'configs')
NO_TRACK_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'no_track')

def load_config(config_name):
    config_path = os.path.join(CONFIGS_DIR, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 없어요: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def train(config):
    if not TEST_RUN:
        wandb_key_path = os.path.join(NO_TRACK_DIR, 'wandb_key.txt')
        with open(wandb_key_path, 'r') as f:
            wandb_key = f.readline().strip()
        wandb.login(key=wandb_key)

        run_time = datetime.now().strftime("%y%m%d_%H%M%S")
        run_name = f"{run_time}_{config['model_size']}"

        # run_name = 'augmentation 실험'
        # config['epochs'] = 3

        wandb.init(
            project=config['task'],
            name=run_name,
            config={
                **config,
                "save_dir": config['save_dir'],
                "run_time": run_time,
                "run_name": run_name,
                "dataset": config['meta_file'],
                "root_dir": config['root_dir']
            }
        )
        config = wandb.config

        # # init 이후에 넣고싶은 내용은 wandb.config.update()로 추가 가능
        # wandb.config.update({
        #     "run_time": run_time,
        #     "run_name": run_name,
        #     })

    if config['task'] == 'age_cls':
        train_dataset, val_dataset = dataloaders.age_cls.get_datasets(config)
        tasks.age_cls.train(train_dataset, val_dataset, config)
    elif config['task'] == 'wrinkle_cls':
        train_dataset, val_dataset = dataloaders.wrinkle_cls.get_datasets(config)
        tasks.wrinkle_cls.train(train_dataset, val_dataset, config)
    elif config['task'] == 'wrinkle_reg':
        train_dataset, val_dataset = dataloaders.wrinkle_reg.get_datasets(config)
        tasks.wrinkle_reg.train(train_dataset, val_dataset, config)
    elif config['task'] == 'pigment_cls':
        train_dataset, val_dataset = dataloaders.pigment_cls.get_datasets(config)
        tasks.pigment_cls.train(train_dataset, val_dataset, config)
    elif config['task'] == 'pigment_reg':
        train_dataset, val_dataset = dataloaders.pigment_reg.get_datasets(config)
        tasks.pigment_reg.train(train_dataset, val_dataset, config)
    elif config['task'] == 'homogenity_cls':
        train_dataset, val_dataset = dataloaders.homogenity_cls.get_datasets(config)
        tasks.homogenity_cls.train(train_dataset, val_dataset, config)
    else:
        raise ValueError(f"Unknown task: {config['task']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config YAML 파일명 (예: pigment_reg.yaml)')
    parser.add_argument('--test', action='store_true', help='TEST_RUN 모드 (wandb 비활성화)')
    args = parser.parse_args()

    if args.test:
        TEST_RUN = True

    config = load_config(args.config)

    try:
        train(config)
        if not TEST_RUN:
            wandb.finish()
        send_slack_message("training finished")

    except Exception as e:
        send_slack_message(f"training failed: {e}")
        raise e