from models.model_params import MODEL_DICT
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Embedding, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def build_model(input_shape=(384, 384, 3), kernel_regularizer=None, model_size='S', num_sectors=7):
    image_input = Input(shape=input_shape, name='image_input')
    sector_input = Input(shape=(1,), dtype=tf.int8, name='sector_input')

    if model_size not in MODEL_DICT.keys():
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_DICT.keys())}.")

    Backbone = MODEL_DICT[model_size](include_top=False, input_shape=input_shape, weights='imagenet')
    Backbone.trainable = True

    kernel_reg = regularizers.l2(kernel_regularizer) if kernel_regularizer else None

    image_features = Backbone(image_input)
    image_features = GlobalAveragePooling2D()(image_features)

    sector_embedding = Embedding(input_dim=num_sectors, output_dim=16, name='sector_embedding')(sector_input)
    sector_embedding = Flatten()(sector_embedding)

    combined_features = Concatenate(name='combined_features')([image_features, sector_embedding])

    shared_dense = Dense(256, activation='relu', name='shared_dense', kernel_regularizer=kernel_reg)(combined_features)
    shared_dense = Dropout(0.3)(shared_dense)

    output = Dense(1, activation=None, name='wrinkle_reg_output')(shared_dense)

    model = Model(inputs=[image_input, sector_input], outputs=output)

    return model, Backbone.name