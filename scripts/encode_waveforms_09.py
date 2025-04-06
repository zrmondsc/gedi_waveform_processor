import os
import numpy as np
from tensorflow.keras.models import load_model

# Set paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "gedi_waveforms_tf.npz")
encoder_path = os.path.join(project_root, "models", "cae_09_encoder.keras")
output_path = os.path.join(project_root, "data", "encoded_latents.npz")

# Load data
data = np.load(data_path)
waveforms = data['waveforms'][..., np.newaxis]  # Shape: (N, 500, 1)
metadata = data['metadata']                    # Shape: (N,) or (N, D)

# Load encoder model
encoder = load_model(encoder_path)

# Encode waveforms
latent_vectors = encoder.predict(waveforms, batch_size=64, verbose=1)

# Save the encoded waveforms and metadata
np.savez(output_path, latents=latent_vectors, metadata=metadata)
print(f"Saved encoded latents + metadata to: {output_path}")
