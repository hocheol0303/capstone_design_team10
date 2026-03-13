from models.model_params import MODEL_DICT
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D 
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
from .common_layers import VChannelEqualizer

def build_model(input_shape=(384, 384, 3), num_classes=7, kernel_regularizer=None, model_size='S'):
    '''
    피부가 아닌 부분은 마스킹처리한 한 장의 이미지를 입력으로 받아서 10대 단위(10대~70대)로 나이를 분류하는 모델
    효율을 위해 백본은 공유하고 성별에 따라 다른 head를 사용한다.
    '''

    inputs = Input(shape=input_shape)

    # EfficientNetV2-S 백본 (input_tensor 대신 input_shape 사용)
    if model_size not in MODEL_DICT.keys():
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_DICT.keys())}.")
    
    Backbone = MODEL_DICT[model_size](include_top=False, input_shape=input_shape, weights='imagenet')
    Backbone.trainable = True  # 전체 fine-tuning 허용

    # x = VChannelEqualizer()(inputs)
    x = Backbone(inputs)
    x = GlobalAveragePooling2D()(x)

    male_head = Dense(128, activation='relu')(x)
    female_head = Dense(128, activation='relu')(x)

    male_output = Dense(num_classes, activation='softmax', name='male', kernel_regularizer=regularizers.l2(kernel_regularizer))(male_head)
    female_output = Dense(num_classes, activation='softmax', name='female', kernel_regularizer=regularizers.l2(kernel_regularizer))(female_head)

    full_model = Model(inputs=inputs, outputs=[male_output, female_output], name='age_cls')

    return full_model, Backbone.name