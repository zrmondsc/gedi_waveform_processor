import numpy as np
import tensorflow as tf
from augment_waveforms_09 import augment_waveform

class SimCLRDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, waveforms, batch_size=64):
        self.waveforms = waveforms
        self.batch_size = batch_size
        self.indices = np.arange(len(waveforms))
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.waveforms) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = self.waveforms[batch_ids]

        x_i = np.array([augment_waveform(w) for w in x])
        x_j = np.array([augment_waveform(w) for w in x])
        return x_i, x_j

    def on_epoch_end(self):
        np.random.shuffle(self.indices)