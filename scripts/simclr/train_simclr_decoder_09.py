import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, Input
from tensorflow.keras.optimizers import Adam

# Set paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, "data", "gedi_waveforms_tf.npz")
encoder_path = os.path.join(project_root, "models", "simclr_encoder_09.keras")
output_path = os.path.join(project_root, "models", "simclr_decoder_09.keras")

# Load data
data = np.load(data_path)
waveforms = data['waveforms'][..., np.newaxis]

# Load and freeze encoder
encoder = load_model(encoder_path)
encoder.trainable = False

# Build new decoder to match encoder output shape
latent_dim = encoder.output_shape[-1]
decoder_input = Input(shape=(latent_dim,), name="decoder_input")

x = layers.Dense(125 * 64, activation='relu')(decoder_input)
x = layers.Reshape((125, 64))(x)
x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
x = layers.UpSampling1D(2)(x)
decoder_output = layers.Conv1D(1, 3, activation='linear', padding='same')(x)

decoder = Model(decoder_input, decoder_output, name="simclr_decoder")

# Build full model: encoder + decoder
inputs = encoder.input
latents = encoder(inputs)
reconstructed = decoder(latents)

autoencoder = Model(inputs, reconstructed, name="simclr_autoencoder")
autoencoder.compile(optimizer=Adam(1e-3), loss='mse')

# Train only the decoder (encoder is frozen)
autoencoder.fit(waveforms, waveforms,
                epochs=20,
                batch_size=64,
                shuffle=True,
                validation_split=0.2,
                verbose=1)

# Save decoder
decoder.save(output_path)
print(f"Decoder saved to {output_path}")