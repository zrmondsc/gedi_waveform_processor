import os
import numpy as np
import tensorflow as tf
from build_cae_09 import build_CAE_09
from build_simclr_model_09 import build_simclr_model
from simclr_generator_09 import SimCLRDataGenerator
from simclr_loss_09 import nt_xent_loss

# Load waveform data
data_path = os.path.join("data", "gedi_waveforms_tf.npz")
data = np.load(data_path)
waveforms = data['waveforms'][..., np.newaxis]

# Build encoder and SimCLR model
_, encoder, _ = build_CAE_09(input_shape=(500, 1))
simclr_model = build_simclr_model(encoder)

# Optimizer
optimizer = tf.keras.optimizers.Adam(1e-3)

# Data generator
train_gen = SimCLRDataGenerator(waveforms, batch_size=128)

# Training loop
epochs = 50
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step in range(len(train_gen)):
        x_i, x_j = train_gen[step]

        with tf.GradientTape() as tape:
            z_i = simclr_model(x_i, training=True)
            z_j = simclr_model(x_j, training=True)
            loss = nt_xent_loss(z_i, z_j)

        grads = tape.gradient(loss, simclr_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, simclr_model.trainable_variables))

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}")

encoder.save("models/simclr_encoder.keras")
