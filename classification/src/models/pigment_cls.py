from models.model_params import MODEL_DICT
from tensorflow.keras import layers # import Input, Dense, GlobalAveragePooling2D 
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from .common_layers import VChannelEqualizer

def build_model(input_shape=(384, 384, 3), num_classes=5, kernel_regularizer=None, model_size='S'):
    '''
    지금은 왼쪽볼, 오른쪽볼 구분 없이 받아서 분류하는 모델
    추후 멀티태스크 모델 구현 시도할 것
    '''
    
    inputs = layers.Input(shape=input_shape)

    # EfficientNetV2-S 백본 (input_tensor 대신 input_shape 사용)
    if model_size not in MODEL_DICT.keys():
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_DICT.keys())}.")
    
    base_model = MODEL_DICT[model_size](include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True  # 전체 fine-tuning 허용

    kernel_reg = regularizers.l2(kernel_regularizer) if kernel_regularizer else None

    # x = VChannelEqualizer()(inputs)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=kernel_reg)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model.name