import os
import gc
import wandb
from wandb.integration.keras import WandbMetricsLogger
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from utils.callbacks import SaveEveryKEpochs, ClearMemoryCallback, WrinkleCallBack
from models.wrinkle_cls import build_model

def get_loss_fn(config):
    if config.loss == 'categorical_crossentropy':
        return tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
    elif config.loss == 'sparse_categorical_crossentropy':
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    elif config.loss == 'categorical_focal_crossentropy':
        return tf.keras.losses.CategoricalFocalCrossentropy(from_logits=False, reduction='none')
    else:
        raise ValueError(f"Unsupported loss function: {config.loss}")

# 학습 loop 직접 구성
def train(train_dataset, val_dataset, config):
    model, model_name = build_model(
        input_shape=config.input_shape,
        kernel_regularizer=config.kernel_regularizer,
        model_size=config.model_size,
        num_classes=5,
        num_sectors=7
    )

    optimizer = Adam(learning_rate=config.learning_rate)
    loss_fn = get_loss_fn(config)
    
    wandb.config.update({"model_name": model_name})
    model.build(input_shape=(None, *config.input_shape))
    # loss나 metric은 직접 계산하는데 optimizer는 callbacks에서 필요로해서 compile 함수로 직접 지정
    model.compile(
        optimizer=optimizer,
        loss=config.loss,
        metrics=[AUC(name='auc', from_logits=False, multi_label=False), 'accuracy']
        )
    model.summary()

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=[
            WandbMetricsLogger(),
            SaveEveryKEpochs(save_dir=os.path.join(config.save_dir, wandb.run.name), k=10, config=config),
            # WrinkleLoggingCallback(config=config),
            WrinkleCallBack(val_dataset=val_dataset, config=config, k=10, num_classes=5, loss_fn=loss_fn),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,         # factor는 다음 lr을 계산할 때 현재 lr에 곱해지는 값
                patience=5,         # patience는 몇 epoch동안 개선이 없을 때 lr을 줄일지 결정
                min_lr=1e-8,
                verbose=1
            ),
            ClearMemoryCallback()
        ]
    )

    del train_dataset
    gc.collect()