from tensorflow.keras.applications import (
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, 
    EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
)

MODEL_DICT = {
    'B0': EfficientNetV2B0,
    'B1': EfficientNetV2B1,
    'B2': EfficientNetV2B2,
    'B3': EfficientNetV2B3,
    'S': EfficientNetV2S,
    'M': EfficientNetV2M,
    'L': EfficientNetV2L,
}