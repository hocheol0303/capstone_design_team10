import tensorflow as tf
from utils.constants import PIGMENT_RANGES

class RangeMSELoss(tf.keras.losses.Loss):
    def __init__(self, name='range_mse_loss'):
        super().__init__(name=name)
        self.ranges = tf.constant(
            [[r[0], r[1]] for r in PIGMENT_RANGES],
            dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        '''
        y_true: GT label (0, 1, 2, 3, 4) - Shape: (batch_size, 1)
        y_pred: Predicted value (0~100) - Shape: (batch_size, 1)
        '''
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.squeeze(y_pred)

        gt_ranges = tf.gather(self.ranges, y_true)
        lower_bounds = gt_ranges[:, 0]
        upper_bounds = gt_ranges[:, 1]

        # lower보다 작은 경우, upper보다 큰 경우에는 패널티 주기
        error_lower = tf.maximum(0.0, lower_bounds - y_pred)
        error_upper = tf.maximum(0.0, y_pred - upper_bounds)

        total_error = error_lower + error_upper
        return tf.reduce_mean(tf.square(total_error))
