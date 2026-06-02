from models.model_params import MODEL_DICT
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def build_model(input_shape=(384, 384, 3), kernel_regularizer=None, model_size='S'):
    '''
    age_cls와 동일한 dual head 구조 (male/female),
    출력은 실제 나이 값 (회귀), 대푯값: 14.5, 24.5, ..., 74.5
    '''
    if model_size not in MODEL_DICT.keys():
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_DICT.keys())}.")

    Backbone = MODEL_DICT[model_size](include_top=False, input_shape=input_shape, weights='imagenet')
    Backbone.trainable = True

    kernel_reg = regularizers.l2(kernel_regularizer) if kernel_regularizer else None

    inputs = Input(shape=input_shape)
    x = Backbone(inputs)
    x = GlobalAveragePooling2D()(x)

    male_head = Dense(128, activation='relu', kernel_regularizer=kernel_reg)(x)
    male_head = Dropout(0.3)(male_head)
    male_output = Dense(1, activation=None, name='male')(male_head)

    female_head = Dense(128, activation='relu', kernel_regularizer=kernel_reg)(x)
    female_head = Dropout(0.3)(female_head)
    female_output = Dense(1, activation=None, name='female')(female_head)

    model = Model(inputs=inputs, outputs=[male_output, female_output], name='age_reg')
    return model, Backbone.name

if __name__ == "__main__":
    model, backbone_name = build_model(model_size='S')
    model.summary()