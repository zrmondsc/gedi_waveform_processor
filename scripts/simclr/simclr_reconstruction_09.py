import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(project_root, "data", "gedi_waveforms_tf.npz")
encoder_path = os.path.join(project_root, "models", "simclr_encoder.keras")
decoder_path = os.path.join(project_root, "models", "cae_09_decoder.keras")

# Load data
data = np.load(data_path)
waveforms = data['waveforms'][..., np.newaxis]  # shape: (N, 500, 1)

# Load models
encoder = load_model(encoder_path)
decoder = load_model(decoder_path)

# Encode and reconstruct
latent = encoder.predict(waveforms[:10], verbose=1)
reconstructed = decoder.predict(latent, verbose=1)

# Plot original vs reconstructed
for i in range(5):
    plt.figure(figsize=(10, 3))
    plt.plot(waveforms[i].squeeze(), label='Original', linewidth=2)
    plt.plot(reconstructed[i].squeeze(), label='SimCLR Reconstruction', linestyle='--')
    plt.title(f'SimCLR Reconstruction - Waveform {i}')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()