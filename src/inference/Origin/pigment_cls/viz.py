# sh run_confusion.sh로 실행할 것!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import re
from math import ceil

# ===== 사용자 설정 =====
INFERENCE_DIR = '/home/work/hocheol_dir/workspace/inference_results/v0.2/pig_3ch/20250701_000901'
CLASS_NAMES = ['0', '1', '2', '3', '4']
CHUNK_SIZE = 10  # 한 번에 처리할 CSV 파일 수
TITLE = "confusion"
# ======================

def plot_confusion_matrix(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)), normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0.0, vmax=1.0, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

def chunk_list(lst, n):
    """리스트를 n개씩 묶어 반환합니다."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_epoch_number(filename):
    match = re.search(r'epoch(\d+)', filename)
    return int(match.group(1)) if match else float('inf')  # 숫자가 없으면 무한대로 설정하여 뒤로 정렬

if __name__ == "__main__":
    # INFERENCE_DIR에서 .csv 파일 목록 가져오기
    csv_files = [f for f in os.listdir(INFERENCE_DIR) if f.endswith('.csv')]
    csv_files.sort(key=extract_epoch_number)  # 파일명 기준 정렬

    # CSV 파일들을 CHUNK_SIZE만큼 묶어서 처리
    for idx, chunk in enumerate(chunk_list(csv_files, CHUNK_SIZE)):
        num_files = len(chunk)
        rows = ceil(num_files / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows))
        axes = axes.flatten()

        for i, csv_name in enumerate(chunk):
            epoch_match = re.search(r'epoch[0-9]+', csv_name)
            epoch = epoch_match.group() if epoch_match else f'file_{i}'
            df = pd.read_csv(os.path.join(INFERENCE_DIR, csv_name))
            y_true = df['label'].values
            y_pred = df['pred'].values
            plot_confusion_matrix(axes[i], y_true, y_pred, title=epoch)

        # 남은 서브플롯 숨기기
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(INFERENCE_DIR, f'{TITLE}_{idx + 1}.png')
        plt.savefig(save_path)
        plt.close(fig)