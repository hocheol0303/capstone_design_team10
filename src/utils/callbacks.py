from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import tensorflow as tf
import wandb
import gc
import os
from datetime import datetime
import shutil
from .tensorflow import predict_step
from .wandb import log_confusion_matrix, log_roc_curve
import math

class AgeClsCallBack(Callback):
    def __init__(self, val_dataset, config, k=10):
        super().__init__()
        self.k = k
        self.val_dataset = val_dataset
        self.config = config
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.k != 0:
            return
        else:
            y_true, y_pred, all_probs = [], [], []

            for batch in tqdm(self.val_dataset, desc=f"Evaluating @ Epoch {epoch+1}"):
                images, labels, weights = batch

                # 메모리 계속 쌓이는 문제 해결하기 위해 predict_step으로 분리해서 @tf.function 적용
                # preds = self.model(images, training=False)
                preds = predict_step(self.model, images)

                for i in range(images.shape[0]):
                    if weights['male'][i].numpy() == 1.0:
                        true = tf.argmax(labels['male'][i]).numpy()
                        pred = tf.argmax(preds[0][i]).numpy()
                        prob = preds[0][i].numpy()
                    else:
                        true = tf.argmax(labels['female'][i]).numpy()
                        pred = tf.argmax(preds[1][i]).numpy()
                        prob = preds[1][i].numpy()
                        
                    y_true.append(true)
                    y_pred.append(pred)
                    all_probs.append(prob)
            
            class_names = ['10', '20', '30', '40', '50', '60', '70']

            confusion_name = f"Confusion_Matrix/{self.config.run_time}_epoch{epoch+1:03d}"
            log_confusion_matrix(y_true, y_pred, epoch, class_names, confusion_name)

            roc_name = f"ROC_Curve/{self.config.run_time}_epoch{epoch+1:03d}"
            log_roc_curve(y_true, all_probs, epoch, class_names, roc_name)

class Pig3chCallBack(Callback):
    def __init__(self, val_dataset, config, k=10):
        super().__init__()
        self.k = k
        self.val_dataset = val_dataset
        self.config = config
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.k != 0:
            return
        else:
            y_true, y_pred, all_probs = [], [], []

            for batch in tqdm(self.val_dataset, desc=f"Evaluating @ Epoch {epoch+1}"):
                images, labels = batch
                
                # 메모리 계속 쌓이는 문제 해결하기 위해 predict_step으로 분리해서 @tf.function 적용
                # preds = self.model(images, training=False)
                preds = predict_step(self.model, images)

                probs = tf.nn.softmax(preds, axis=-1)
                pred_labels = tf.argmax(probs, axis=-1)

                # one-hot encoding일 경우 정수 클래스로 변환
                if len(labels.shape) > 1 and labels.shape[-1] > 1:
                    labels = tf.argmax(labels, axis=-1)

                all_probs.extend(probs.numpy())
                y_true.extend(labels.numpy().tolist())
                y_pred.extend(pred_labels.numpy().tolist())
            
            class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']

            confusion_name = f"Confusion_Matrix/{self.config.run_time}_epoch{epoch+1:03d}"
            log_confusion_matrix(y_true, y_pred, epoch, class_names, confusion_name)

            roc_name = f"ROC_Curve/{self.config.run_time}_epoch{epoch+1:03d}"
            log_roc_curve(y_true, all_probs, epoch, class_names, roc_name)

class WrinkleCallBack(Callback):
    # callback이 모델에 연결되면 self.model은 공짜
    def __init__(self, val_dataset, config, k=10, num_classes=5, loss_fn=None):
        super().__init__()
        self.k = k
        self.val_dataset = val_dataset
        self.config = config
        self.num_classes = num_classes
        self.loss_fn = loss_fn

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.k != 0:
            return
        else:
            y_true, y_pred, all_probs = [], [], []

            for batch in tqdm(self.val_dataset, desc=f"Evaluating @ Epoch {epoch+1}"):
                inputs, labels = batch

                # 메모리 계속 쌓이는 문제 해결하기 위해 predict_step으로 분리해서 @tf.function 적용
                # wrinkle_preds = self.model(inputs, training=False)
                wrinkle_preds = predict_step(self.model, inputs)
                
                # dataloader가 dataset 만들 때 모든 부위에 정답 label을 복사해뒀기 때문에 어떤 부위를 써도 상관 없음
                y_true.extend(tf.argmax(labels, axis=1).numpy().tolist())
                y_pred.extend(tf.argmax(wrinkle_preds, axis=1).numpy().tolist())
                all_probs.extend(wrinkle_preds.numpy().tolist())
            
            class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
            confusion_name = f"Confusion_Matrix/{self.config.run_time}_epoch{epoch+1:03d}"
            log_confusion_matrix(y_true, y_pred, epoch, class_names, confusion_name)
            
            roc_name = f"ROC_Curve/{self.config.run_time}_epoch{epoch+1:03d}"
            log_roc_curve(y_true, all_probs, epoch, class_names, roc_name)

class WrinkleLoggingCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        for key, value in logs.items():
            if key in ['epoch', 'val_loss', 'train_loss', 'lr']:
                wandb.log({f"epoch/{key}": value}, step=epoch)
            elif 'loss' in key:
                wandb.log({f"SectorLoss/{key}": value}, step=epoch)
            elif 'auc' in key:
                wandb.log({f"SectorAUC/{key}": value}, step=epoch)
            else:
                wandb.log({f"Other/{key}": value}, step=epoch)
        print(f"Epoch {epoch}:")
        for key, value in logs.items():
            print(f"\t{key}: {value:.4f}")

class ClearMemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # tf.keras.backend.clear_session()은 메모리 오버헤드가 생길 수 있음
        # K.clear_session()
        gc.collect()
        # reset_memory_stats는 GPU 메모리 통계 카운터를 리셋하는 것 (실제 메모리 해제 아님)
        tf.config.experimental.reset_memory_stats('GPU:0')
        print(f"[Epoch {epoch+1}] gc.collect 완료")

class SaveTopKModels(Callback):
    def __init__(self, k=3, save_dir="saved_models", time=None):
        super().__init__()
        self.k = k
        self.save_dir = os.path.join(save_dir, time)
        os.makedirs(self.save_dir, exist_ok=True)
        self.saved_models = []  # (filepath, acc)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 != 0: # 5의 배수일 때에는 저장하는 코드 따로 있음
            loss = logs.get("val_loss", 0.0)
            now = datetime.now().strftime("%y%m%d_%H%M%S")
            filename = f"epoch{epoch+1}_{now}_loss{loss:.4f}.keras"
            filepath = os.path.join(self.save_dir, filename)

            # 현재 모델 저장
            self.model.save(filepath)
            self.saved_models.append((filepath, loss))
            print(f"✅ [Epoch {epoch+1}] 모델 저장 완료 → {filepath}")

            # loss 기준으로 상위 k개만 유지
            self.saved_models.sort(key=lambda x: x[1], reverse=False) # 오름차순 정렬
            if len(self.saved_models) > self.k:
                to_delete = self.saved_models[self.k:]
                for path, _ in to_delete:
                    try:
                        os.remove(path)
                        print(f"🗑️ 제거된 모델: {path}")
                    except:
                        print(f"⚠️ 삭제 실패: {path}")
                self.saved_models = self.saved_models[:self.k]

class SaveEveryKEpochs(Callback):
    def __init__(self, save_dir="saved_models", k=5, config=None):
        super().__init__()
        self.config = config
        self.k = k
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("val_loss", 0.0)

        # k 에폭마다 저장
        if (epoch + 1) % self.k == 0:
            filename = f"epoch{epoch+1}_{self.config.run_time}.keras"
            filepath = os.path.join(self.save_dir, filename)
            artifact_name = f"{self.config.run_name}-model"

            temp_path = os.path.join(self.save_dir, f"temp_dir", filename)

            try:
                # temp에 저장
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                self.model.save(temp_path)
                print(f"📝 [Epoch {epoch+1}] 임시 모델 저장 완료 → {temp_path}")

                # 로컬에 저장
                shutil.copy(temp_path, filepath)
                print(f"✅ [Epoch {epoch+1}] 모델 저장 완료 → {filepath}")

                # wandb에 저장
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type='model',
                    description=f"Model from epoch {epoch+1} of run {self.config.run_name}",
                    metadata={'epoch': epoch+1, **logs}
                )
                artifact.add_file(local_path=temp_path)
                wandb.log_artifact(artifact, aliases=[f"epoch{epoch+1}"])
                print(f"📦 [Epoch {epoch+1}] 모델 아티팩트로 저장됨 → {filename}")
            except Exception as e:
                print(f"⚠️ [Epoch {epoch+1}] 모델 저장 실패: {e}")
            finally:
                if os.path.exists(temp_path):
                    # 임시 디렉토리 삭제
                    os.remove(temp_path)
                    print(f"🗑️ 임시 파일 삭제 완료: {temp_path}")

class CosineAnnealingCallback(Callback):
    def __init__(self, epochs, cycles, lr_max, min_lr):
        super().__init__()
        self.epochs_per_cycle = math.floor(epochs / cycles)
        self.lr_max = lr_max
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        cos_inner = (math.pi * (epoch % self.epochs_per_cycle)) / self.epochs_per_cycle
        lr = self.min_lr + (self.lr_max - self.min_lr) / 2 * (math.cos(cos_inner) + 1)
        self.model.optimizer.learning_rate.assign(lr)
        print(f"\n\033[47mEpoch {epoch + 1:03d}: Learning rate is {lr:.6f}\033[0m")