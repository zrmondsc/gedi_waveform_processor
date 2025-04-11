from tensorflow.keras import layers, models, optimizers, Input

def build_CAE_09(input_shape=(500, 1)):
    """
    Build CAE_09: Best-performing convolutional autoencoder from experiment #9.

    Returns:
        autoencoder (Model): Full autoencoder model (encoder + decoder)
        encoder (Model): Encoder-only model
        decoder (Model): Decoder-only model
    """
    # Encoder
    inputs = Input(shape=input_shape, name='input_layer')
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(8, activation='linear', name='bottleneck')(x)

    encoder = models.Model(inputs, bottleneck, name="encoder")

    # Decoder
    decoder_input = Input(shape=(8,), name='decoder_input')
    x = layers.Dense(125 * 64, activation='relu')(decoder_input)
    x = layers.Reshape((125, 64))(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    decoder_output = layers.Conv1D(1, 3, activation='linear', padding='same')(x)

    decoder = models.Model(decoder_input, decoder_output, name="decoder")

    # Autoencoder model
    autoencoder_output = decoder(encoder(inputs))
    autoencoder = models.Model(inputs, autoencoder_output, name="autoencoder")
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

    return autoencoder, encoder, decoder
