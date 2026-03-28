from models.model_params import MODEL_DICT
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers

def build_model(input_shape=(384, 384, 3), kernel_regularizer=None, model_size='S'):
    if model_size not in MODEL_DICT.keys():
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_DICT.keys())}.")
    
    base_model = MODEL_DICT[model_size](include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True

    kernel_reg = regularizers.l2(kernel_regularizer) if kernel_regularizer else None

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=kernel_reg)(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation=None)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model, base_model.name