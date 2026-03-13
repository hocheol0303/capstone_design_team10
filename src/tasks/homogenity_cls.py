import os
import wandb
import gc
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.optimizers import Adam
from models.homogenity_cls import build_model
from tensorflow.keras.metrics import AUC
from utils.callbacks import SaveEveryKEpochs, ClearMemoryCallback, CosineAnnealingCallback
from datetime import datetime


def train(train_dataset, val_dataset, config):
    print("\033[44m모델 생성 중...\033[0m")
    model, model_name = build_model(
        input_shape=config.input_shape,
        kernel_regularizer=config.kernel_regularizer,
        model_size=config.model_size
    )

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss={ # 다중 output에 맞게 loss도 dict 형태로 전달
            'rad_out': config.loss,
            'tex_out': config.loss
        },
        metrics=['accuracy', AUC(name='auc', from_logits=False, multi_label=False)]
    )

    wandb.config.update({'model_name': model_name})
    model.build((None, *config.input_shape))
    model.summary()

    start_time = datetime.now().strftime('%y%m%d_%H%M%S')

    print('\033[44m학습 시작...\033[0m')
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=[
            WandbMetricsLogger(),
            SaveEveryKEpochs(save_dir=os.path.join(config.save_dir, wandb.run.name), k=10, config=config),
            CosineAnnealingCallback(epochs=config.epochs, cycles=1, lr_max=config.learning_rate, min_lr=1e-6),
            ClearMemoryCallback()
        ]
    )

    end_time = datetime.now().strftime('%y%m%d_%H%M%S')
    print(f"\n\033[42m학습 완료!\033[0m\n시작 시간: {start_time}\n종료 시간: {end_time}")
    del train_dataset
    gc.collect()
