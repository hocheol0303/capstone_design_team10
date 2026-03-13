import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from math import ceil

INFERENCE_DIR = '/home/work/hocheol_dir/workspace/inference_results/age_only1_20250529_091414'
CHUNK_SIZE = 10

# 빨간선 아래가 어리게 예측
def plot_predicted_vs_actual(y_true, y_pred, ax, title):
    ax.scatter(y_true, y_pred, alpha=0.5, label='pred')
    ax.plot([0, 100],
             [0, 100],
             'r--', label='perfect_pred (y = x)')
    ax.set_xlabel('true_age')
    ax.set_ylabel('pred_age')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True)
    # plt.show()

# 빨간선 위는 어리게 예측
def plot_residuals(y_true, y_pred, ax, title):
    """
    Plot residuals (actual - predicted) against predicted values.

    Parameters:
    - y_true: Actual values (array-like)
    - y_pred: Predicted values (array-like)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(-50, 50)
    ax.grid(True)
    # plt.show()

if __name__ == "__main__":
    # .csv 파일 목록 가져오기
    csv_files = [f for f in os.listdir(INFERENCE_DIR) if f.endswith('.csv')]
    csv_files.sort(key=lambda x:int(re.search(r'epoch(\d+)', x).group(1)))
    

    num_chunks = ceil(len(csv_files) / CHUNK_SIZE)
    csv_chunks = [csv_files[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(num_chunks)]

    for idx, chunk in enumerate(csv_chunks):
        fig, axes = plt.subplots(len(chunk), 2, figsize=(10, 5 * len(chunk)))
        if len(chunk) == 1:
            axes = [axes]  # axes를 2차원 리스트로 변환

        for i, csv_name in enumerate(chunk):
            epoch_match = re.search(r'epoch[0-9]+', csv_name)
            epoch = epoch_match.group() if epoch_match else f'file_{i}'

            df = pd.read_csv(os.path.join(INFERENCE_DIR, csv_name))
            y_true = df['true_age'].to_numpy()
            y_pred = df['pred_age'].to_numpy()

            plot_predicted_vs_actual(y_true, y_pred, axes[i][0], f'{epoch} - Predicted vs Actual')
            plot_residuals(y_true, y_pred, axes[i][1], f'{epoch} - Residuals')

        plt.tight_layout()
        save_path = os.path.join(INFERENCE_DIR, f'plots_batch_{idx+1}.png')
        plt.savefig(save_path)
        plt.close(fig)
