import numpy as np

def add_gaussian_noise(x, std=0.01):
    return x + np.random.normal(0, std, size=x.shape)

def random_crop_and_resize(x, crop_fraction=0.9):
    crop_len = int(len(x) * crop_fraction)
    start = np.random.randint(0, len(x) - crop_len)
    cropped = x[start:start+crop_len]
    return np.interp(np.linspace(0, crop_len, len(x)), np.arange(crop_len), cropped)

def time_mask(x, mask_fraction=0.1):
    x = x.copy()
    mask_len = int(len(x) * mask_fraction)
    start = np.random.randint(0, len(x) - mask_len)
    x[start:start+mask_len] = 0
    return x

def amplitude_jitter(x, scale=0.1):
    return x * (1 + np.random.uniform(-scale, scale, size=x.shape))

def augment_waveform(x):
    x = x.squeeze()
    aug = np.random.choice([add_gaussian_noise, random_crop_and_resize, time_mask, amplitude_jitter])
    x_aug = aug(x)
    return x_aug[..., np.newaxis]