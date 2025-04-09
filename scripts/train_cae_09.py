import os
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from build_cae_09 import build_CAE_09

# Set paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "gedi_waveforms_tf.npz")
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True)

# Load data
data = np.load(data_path)
waveforms = data['waveforms'][..., np.newaxis]
metadata = data['metadata']
shot_index = data['shot_index']

# Split and shuffle data
x_train, x_val, meta_train, meta_val, si_train, si_val = train_test_split(
    waveforms, metadata, shot_index, test_size=0.3, shuffle=True, random_state=42
)

# Build and initialize the model
input_shape = x_train.shape[1:]
autoencoder, encoder, decoder = build_CAE_09(input_shape)

# Callbacks
log_dir = os.path.join(project_root, "logs", datetime.datetime.now().strftime("fit-%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model
history = autoencoder.fit(
    x_train, x_train,
    epochs=100,
    batch_size=64,
    validation_data=(x_val, x_val),
    callbacks=[tensorboard_callback, early_stop]
)

# Save the model and model components
autoencoder.save(os.path.join(model_dir, "cae_09_autoencoder.keras"))
encoder.save(os.path.join(model_dir, "cae_09_encoder.keras"))
decoder.save(os.path.join(model_dir, "cae_09_decoder.keras"))