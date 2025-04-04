import numpy as np
import ast
from scipy.signal import resample, resample_poly

def decode_waveform(wf_input):
    """
    Single-waveform utility. Decodes waveform from a list or a string. Handles strings like '1.2, 3.4, 5.6' 
    by wrapping in brackets first.
    """
    try:
        if isinstance(wf_input, list):
            return wf_input

        if isinstance(wf_input, str):
            wf_str = wf_input.strip()

            # If the string doesn't start with a bracket, wrap it as a list
            if not wf_str.startswith("["):
                wf_str = "[" + wf_str + "]"

            return ast.literal_eval(wf_str)

        print(f"Unexpected type for waveform: {type(wf_input)} — skipping row")
        return None

    except Exception as e:
        print(f"Exception while decoding waveform: {repr(wf_input)[:50]} — {type(e).__name__}: {e}")
        return None

def resample_waveform(waveform, target_length=500, method='polyphase'):
    """
    Single-waveform utility. Resamples the waveform to a fixed number of bins.

    Parameters:
        waveform (np.ndarray): Original waveform.
        target_length (int): Desired length after resampling.
        method (str): Resampling method ('polyphase', 'fft', 'pad', 'truncate').

    Returns:
        np.ndarray: Resampled waveform.
    """
    original_length = len(waveform)
    if method == 'polyphase':
        return resample_poly(waveform, target_length, original_length)
    elif method == 'fft':
        return resample(waveform, target_length)
    elif method == 'pad':
        return np.pad(waveform, (0, max(0, target_length - original_length)), mode='edge')[:target_length]
    elif method == 'truncate':
        return waveform[:target_length]
    else:
        raise ValueError(f"Unsupported resampling method: {method}")

def normalize_waveform(waveform, method='zscore'):
    """
    Single-waveform utility. Normalizes a waveform using z-score or min-max normalization.

    Parameters:
        waveform (np.ndarray): Input waveform.
        method (str): 'zscore' or 'minmax'.

    Returns:
        np.ndarray: Normalized waveform.
    """
    if method == 'minmax':
        return (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-8)
    elif method == 'zscore':
        return (waveform - waveform.mean()) / (waveform.std() + 1e-8)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def process_waveform(rxwaveform, target_length=500, resample_method='polyphase', norm_method='zscore'):
    """
    Single-waveform utility. Decodes, resamples, and normalizes a waveform.

    Parameters:
        rxwaveform (list or str): Raw waveform.
        target_length (int): user-defined length of waveform.
        resample_method (str): Resampling strategy.
        norm_method (str): Normalization strategy.

    Returns:
        np.ndarray: Processed waveform.
    """
    decoded = decode_waveform(rxwaveform)
    resampled = resample_waveform(decoded, target_length, resample_method)
    normalized = normalize_waveform(resampled, norm_method)
    return normalized
