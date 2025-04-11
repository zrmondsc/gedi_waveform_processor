import tensorflow as tf
from tensorflow.keras import layers, models

def build_simclr_model(base_encoder, projection_dim=32):
    inputs = base_encoder.input
    features = base_encoder.output

    x = layers.Dense(128, activation='relu')(features)
    x = layers.Dense(projection_dim, activation=None)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                  output_shape=lambda input_shape: input_shape)(x)

    model = models.Model(inputs, x, name='simclr_model')
    return model