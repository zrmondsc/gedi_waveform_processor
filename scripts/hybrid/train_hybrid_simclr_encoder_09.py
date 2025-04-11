import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers, Input
from tensorflow.keras.optimizers import Adam

# Parameters
lambda_recon = 1.0  # Weight of reconstruction loss
projection_dim = 32
latent_dim = 8

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, "data", "gedi_waveforms_tf.npz")
output_encoder = os.path.join(project_root, "models", "hybrid_simclr_encoder_09.keras")
output_decoder = os.path.join(project_root, "models", "hybrid_simclr_decoder_09.keras")

# Load data
data = np.load(data_path)
waveforms = data['waveforms'][..., np.newaxis]

# ----- Build Encoder -----
inputs = Input(shape=(500, 1), name="waveform_input")
x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
x = layers.MaxPooling1D(2, padding='same')(x)
x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling1D(2, padding='same')(x)
x = layers.Flatten()(x)
latent = layers.Dense(latent_dim, activation='linear', name='latent')(x)

# Encoder model
encoder = Model(inputs, latent, name="encoder")

# ----- Projection Head -----
proj = layers.Dense(128, activation='relu')(latent)
proj = layers.Dense(projection_dim)(proj)
proj = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                     output_shape=lambda s: s)(proj)

# ----- Decoder -----
decoder_input = Input(shape=(latent_dim,))
y = layers.Dense(125 * 64, activation='relu')(decoder_input)
y = layers.Reshape((125, 64))(y)
y = layers.Conv1D(64, 3, padding='same', activation='relu')(y)
y = layers.UpSampling1D(2)(y)
y = layers.Conv1D(32, 3, padding='same', activation='relu')(y)
y = layers.UpSampling1D(2)(y)
reconstructed = layers.Conv1D(1, 3, padding='same', activation='linear')(y)

decoder = Model(decoder_input, reconstructed, name="decoder")

# ----- Build Hybrid Model -----
# Full model outputs: [projection, reconstruction]
projection = proj
recon_output = decoder(latent)

hybrid_model = Model(inputs, [projection, recon_output])

# ----- Contrastive Loss -----
def nt_xent_loss(z_i, z_j, temperature=0.1):
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    batch_size = tf.shape(z_i)[0]
    z = tf.concat([z_i, z_j], axis=0)
    sim_matrix = tf.matmul(z, z, transpose_b=True)
    sim_matrix /= temperature
    logits_mask = tf.ones_like(sim_matrix) - tf.eye(2 * batch_size)
    logits = sim_matrix * logits_mask - 1e9 * tf.eye(2 * batch_size)
    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)
    labels = tf.cast(labels, tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)

# ----- Training Loop -----
optimizer = Adam(1e-3)
batch_size = 128
epochs = 50

# Training step
@tf.function
def train_step(x_i, x_j):
    with tf.GradientTape() as tape:
        z_i, x_i_recon = hybrid_model(x_i, training=True)
        z_j, x_j_recon = hybrid_model(x_j, training=True)
        contrastive_loss = nt_xent_loss(z_i, z_j)
        recon_loss = tf.reduce_mean(tf.square(x_i - x_i_recon) + tf.square(x_j - x_j_recon))
        total_loss = contrastive_loss + lambda_recon * recon_loss
    grads = tape.gradient(total_loss, hybrid_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, hybrid_model.trainable_variables))
    return contrastive_loss, recon_loss, total_loss

# ----- Data Generator with Augmentations -----
from augment_waveforms_09 import augment_waveform

def make_augmented_batch(waveforms, batch_size):
    idx = np.random.choice(len(waveforms), batch_size, replace=False)
    x = waveforms[idx]
    x_i = np.array([augment_waveform(w) for w in x]).astype(np.float32)
    x_j = np.array([augment_waveform(w) for w in x]).astype(np.float32)
    return x_i, x_j


# ----- Training Loop -----
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    steps_per_epoch = len(waveforms) // batch_size
    for step in range(steps_per_epoch):
        x_i, x_j = make_augmented_batch(waveforms, batch_size)
        c_loss, r_loss, t_loss = train_step(x_i, x_j)
        if step % 10 == 0:
            print(f"Step {step}, Contrastive: {c_loss:.4f}, Recon: {r_loss:.4f}, Total: {t_loss:.4f}")

# Save encoder and decoder
encoder.save(output_encoder)
decoder.save(output_decoder)
print(f"Saved hybrid encoder to: {output_encoder}")
print(f"Saved hybrid decoder to: {output_decoder}")