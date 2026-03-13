from tensorflow.keras.layers import Layer
import tensorflow as tf

class VChannelEqualizer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # V 채널에 적용할 학습 가능한 scale, bias 파라미터
        self.v_scale = self.add_weight(
            name='v_scale',
            shape=(1, 1, 1),
            initializer='ones',
            trainable=True
        )
        self.v_bias = self.add_weight(
            name='v_bias',
            shape=(1, 1, 1),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        hsv = tf.image.rgb_to_hsv(inputs)
        h, s, v = tf.split(hsv, 3, axis=-1)
        # 학습 가능한 scale, bias 적용
        v_eq = tf.clip_by_value(self.v_scale * v + self.v_bias, 0, 1)
        hsv_eq = tf.concat([h, s, v_eq], axis=-1)
        rgb_eq = tf.image.hsv_to_rgb(hsv_eq)
        return rgb_eq
