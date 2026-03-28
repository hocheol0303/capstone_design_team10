import tensorflow as tf
from utils.constants import PIGMENT_RANGES

class RangeMetrics:
    def __init__(self):
        self.ranges = tf.constant(PIGMENT_RANGES, dtype=tf.float32)
    
    def midpoint_mae(self, y_true, y_pred):
        '''
        y_true: GT label (0, 1, 2, 3, 4) - Shape: (batch_size, 1)
        y_pred: Predicted value (0~100) - Shape: (batch_size, 1)
        '''
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.squeeze(y_pred)

        mid_points = tf.gather(self.ranges[:, 2], y_true)
        mae = tf.reduce_mean(tf.abs(y_pred - mid_points))
        return mae
    
    def range_accuracy(self, y_true, y_pred):
        '''
        y_true: GT label (0, 1, 2, 3, 4) - Shape: (batch_size, 1)
        y_pred: Predicted value (0~100) - Shape: (batch_size, 1)
        '''
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.squeeze(y_pred)

        gt_ranges = tf.gather(self.ranges, y_true)
        lower_bounds = gt_ranges[:, 0]
        upper_bounds = gt_ranges[:, 1]

        in_range = tf.logical_and(y_pred >= lower_bounds, y_pred <= upper_bounds)
        accuracy = tf.reduce_mean(tf.cast(in_range, tf.float32))
        return accuracy