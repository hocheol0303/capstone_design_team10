import os
import gc
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras import callbacks
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from models.pigment_cls import build_model
from utils.callbacks import SaveTopKModels, SaveEveryKEpochs, ClearMemoryCallback, Pig3chCallBack, CosineAnnealingCallback
from tqdm import tqdm

def train(train_dataset, val_dataset, config):
    model, model_name = build_model(
        input_shape=config.input_shape,
        kernel_regularizer=config.kernel_regularizer,
        model_size=config.model_size
    )

    run_time = datetime.fromtimestamp(wandb.run.start_time).strftime("%y%m%d_%H%M%S")

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss=config.loss,
        metrics=[
            'accuracy',
            AUC(name='auc', from_logits=False, multi_label=False)
        ]
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
            SaveEveryKEpochs(save_dir=os.path.join(config.save_dir, wandb.run.name), k=10, config=config),
            Pig3chCallBack(val_dataset=val_dataset, config=config, k=10),
            CosineAnnealingCallback(epochs=config.epochs, cycles=1, lr_max=config.learning_rate, min_lr=1e-6),
            ClearMemoryCallback()
        ],
    )

    end_time = datetime.now().strftime("%y%m%d_%H%M%S")
    print(f'start time : {start_time}\nend time : {end_time}')
    
    del train_dataset
    gc.collect()
