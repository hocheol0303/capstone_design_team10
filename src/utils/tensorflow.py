import tensorflow as tf

@tf.function
def predict_step(model, inputs):
    return model(inputs, training=False)