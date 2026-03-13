import numpy as np
import os
import tensorflow as tf

class HistogramEqualizer:
    def __init__(self, dataset_dirs: list):
        global_values = np.zeros((256,), dtype=np.int64)

        for path in dataset_dirs:
            value_path = os.path.join(path, 'hsv_v_values.npy')
            v_values = np.load(value_path)
            global_values += v_values
        
        # 0 ~ 255 픽셀 값에 대한 확률 분포 계산(각 값의 빈도 수 / 전체 픽셀 수)
        pdf = global_values / global_values.sum()

        # 누적 분포 함수 계산
        self.cdf_np = np.cumsum(pdf).astype(np.float32)
        self.cdf_tf = tf.constant(self.cdf_np, dtype=tf.float32)

    # np 이미지를 받아서 쓸 때 사용
    def get_normalized_image(self, image_rgb: np.ndarray):
        import cv2
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        v_channel = image_hsv[:, :, 2]
        v_equalized = (255*self.cdf_np[v_channel]).astype(np.uint8)
        image_hsv[:, :, 2] = v_equalized
        equalized_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        return equalized_rgb