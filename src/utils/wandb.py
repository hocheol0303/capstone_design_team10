import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def log_confusion_matrix(y_true, y_pred, epoch, class_names, confusion_name):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0.0, vmax=1.0, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"Confusion_epoch{epoch+1:03d}")
    wandb.log({confusion_name: wandb.Image(fig)})
    plt.close(fig)

def log_roc_curve(y_true, all_probs, epoch, class_names, roc_name):
    wandb.log({
        roc_name: wandb.plot.roc_curve(
            y_true,
            np.array(all_probs),
            labels=class_names,
            title=f"ROC_epoch{epoch+1:03d}",
            split_table=True
        )
    })

def load_wandb_model(artifact_name):
    '''
    Args:
        artifact_name: entity/project_name/artifact_name
    
    Returns:
        tf.keras.Model
    '''
    import os
    import tensorflow as tf
    import shutil
    try:
        run = wandb.init(job_type='load_model')
        artifact = run.use_artifact(artifact_name)

        artifact_dir = artifact.download()
        print(f"Artifact {artifact_name} downloaded to {artifact_dir}")

        model_file_path = None
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                if file.endswith('.keras'):
                    model_file_path = os.path.join(root, file)
            if model_file_path:
                break
        if not model_file_path:
            raise ValueError(f"No model file found in artifact {artifact_name}")
        else:
            model = tf.keras.models.load_model(model_file_path)
            shutil.rmtree(artifact_dir)
        
            return model
    except Exception as e:
        raise e
    finally:
        wandb.finish()
