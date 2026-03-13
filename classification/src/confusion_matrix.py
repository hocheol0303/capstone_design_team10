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
INFERENCE_DIR = '/home/work/hocheol_dir/workspace/inference_results/pigmentation_20250522_083700'
CLASS_NAMES = ['0', '1', '2', '3', '4']
CHUNK_SIZE = 10  # 한 번에 처리할 CSV 파일 수
TITLE = "CE"
# ======================

def relaxed_pred(y_true, y_pred, margin=1):
    """
    y_true와 y_pred를 받아서, margin 안에 드는 값으로 y_pred를 수정한다.
    GT와의 차이가 margin을 넘으면 원래 pred 유지 (혹은 가장 가까운 경계로 클립 가능)
    """
    relaxed = []
    for t, p in zip(y_true, y_pred):
        if abs(t - p) <= margin:
            relaxed.append(t)  # 정답 처리
        else:
            relaxed.append(p)  # 그대로 유지하거나 t에 가까운 경계로 조정할 수도 있음
    return relaxed

def plot_confusion_matrix(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
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

# relaxed_pred를 사용한 confusion matrix 생성
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
            y_pred_relaxed = relaxed_pred(y_true, y_pred, margin=1)
            plot_confusion_matrix(axes[i], y_true, y_pred_relaxed, title=epoch)

        # 남은 서브플롯 숨기기
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(INFERENCE_DIR, f'{TITLE}_{idx + 1}_relaxed.png')
        plt.savefig(save_path)
        plt.close(fig)
