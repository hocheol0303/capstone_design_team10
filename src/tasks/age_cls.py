import os
import gc
import wandb
from wandb.integration.keras import WandbMetricsLogger
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from utils.callbacks import SaveTopKModels, SaveEveryKEpochs, ClearMemoryCallback, AgeClsCallBack
from tqdm import tqdm
from models.age_cls import build_model
from tensorflow.keras.metrics import AUC

def train(train_dataset, val_dataset, config):
    model, model_name = build_model(
        input_shape=config.input_shape,
        kernel_regularizer=config.kernel_regularizer,
        model_size=config.model_size
    )

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss={
            'male': config.loss,
            'female': config.loss
        },
        metrics={
            'male': ['accuracy', AUC(name='auc', from_logits=False, multi_label=False)],
            'female': ['accuracy', AUC(name='auc', from_logits=False, multi_label=False)]
        }
    )

    wandb.config.update({"model_name": model_name})
    model.build((None, *config.input_shape[:2], config.input_shape[2]))
    model.summary()

    start_time = datetime.now().strftime("%y%m%d_%H%M%S")
    
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=[
            WandbMetricsLogger(),
            # SaveTopKModels(save_dir=os.path.join(config.save_dir, wandb.run.name), time=run_time),
            SaveEveryKEpochs(save_dir=os.path.join(config.save_dir, wandb.run.name), k=10, config=config),
            AgeClsCallBack(val_dataset=val_dataset, config=config, k=10),
            callbacks.ReduceLROnPlateau(
                monitor='val_female_auc',
                factor=0.7,         # factor는 다음 lr을 계산할 때 현재 lr에 곱해지는 값
                patience=5,         # patience는 몇 epoch동안 개선이 없을 때 lr을 줄일지 결정
                min_lr=1e-6,
                verbose=1
            ),
            ClearMemoryCallback()
        ],
    )

    end_time = datetime.now().strftime("%y%m%d_%H%M%S")
    print(f'start time : {start_time}\nend time : {end_time}')
    
    del train_dataset
    gc.collect()