from models.model_params import MODEL_DICT
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers

def build_model(input_shape=(384, 384, 3), num_classes=10, kernel_regularizer=None, model_size='S'):
    if model_size not in MODEL_DICT.keys():
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_DICT.keys())}.")
    
    base_model = MODEL_DICT[model_size](include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True

    kernel_reg = regularizers.l2(kernel_regularizer) if kernel_regularizer else None

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    # 실험할 것: Dropout vs kernel_regularizer
    x = layers.Dropout(0.3)(x)
    
    # output name들이 DataLoader에서 사용하는 label의 이름이 됨
    rad_output = layers.Dense(num_classes, activation="softmax", name='rad_out', kernel_regularizer=kernel_reg)(x)
    tex_output = layers.Dense(num_classes, activation="softmax", name='tex_out', kernel_regularizer=kernel_reg)(x)

    model = models.Model(inputs=inputs, outputs=[rad_output, tex_output])
    return model, base_model.name