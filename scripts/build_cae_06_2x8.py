from tensorflow.keras import layers, models, optimizers, Input

def build_CAE_06_2x8(input_shape=(500, 1), dropout_rate=0.1):
    """
    Build a convolutional autoencoder with structured latent space (latent_len, latent_dim).
    """

    latent_len, latent_dim = (4, 4)
    latent_size = latent_len * latent_dim

    # Encoder
    inputs = layers.Input(shape=input_shape, name='input_layer')

    x = layers.Conv1D(32, 3, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)

    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)  # Output: (125, 64)

    x = layers.Flatten()(x)  # shape: (125 * 64 = 8000)


    x = layers.Dropout(dropout_rate)(x)

    # Bottleneck
    bottleneck = layers.Dense(latent_size, activation='linear', name='bottleneck')(x)
    reshaped_bottleneck = layers.Reshape((latent_len, latent_dim), name='latent_reshape')(bottleneck)

    encoder = models.Model(inputs, reshaped_bottleneck, name='encoder')

    # Decoder
    decoder_input = layers.Input(shape=(latent_len, latent_dim), name='decoder_input')
    x = layers.Flatten()(decoder_input)  # shape: (latent_len * latent_dim,)
    x = layers.Dense(125 * 64, activation='relu')(x)
    x = layers.Reshape((125, 64))(x)

    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling1D(2)(x)

    x = layers.Conv1D(32, 3, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling1D(2)(x)

    decoded = layers.Conv1D(1, 3, padding='same', activation='linear')(x)

    decoder = models.Model(decoder_input, decoded, name='decoder')

    # Autoencoder
    autoencoder_output = decoder(encoder(inputs))
    autoencoder = models.Model(inputs, autoencoder_output, name='autoencoder')
    autoencoder.compile(optimizer=optimizers.Adam(1e-3), loss='mse')

    return autoencoder, encoder, decoder
