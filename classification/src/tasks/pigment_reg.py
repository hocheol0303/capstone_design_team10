import os
import wandb
import gc
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.optimizers import Adam
from models.pigment_reg import build_model
from custom_losses.range_mse_loss import RangeMSELoss
from utils.callbacks import SaveEveryKEpochs, ClearMemoryCallback, CosineAnnealingCallback
from datetime import datetime
from utils.metrics import RangeMetrics


def train(train_dataset, val_dataset, config):
    print("\033[44m모델 생성 중...\033[0m")
    model, model_name = build_model(
        input_shape=config.input_shape,
        kernel_regularizer=config.kernel_regularizer,
        model_size=config.model_size
    )

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss=RangeMSELoss(),
        metrics=[
            RangeMetrics().midpoint_mae,
            RangeMetrics().range_accuracy
        ]
    )

    wandb.config.update({"model_name": model_name})

    model.build((None, *config.input_shape))
    model.summary()

    start_time = datetime.now().strftime("%y%m%d_%H%M%S")

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

    end_time = datetime.now().strftime("%y%m%d_%H%M%S")
    print(f"\n\033[45mstart time : {start_time}\nend time : {end_time}\033[0m]")
    del train_dataset
    gc.collect()
