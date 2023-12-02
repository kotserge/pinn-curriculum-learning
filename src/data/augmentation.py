import numpy as np


def augment_by_noise(y_train: np.ndarray, snr: float) -> np.ndarray:
    """Adds noise to the output of the PDE.

    Args:
        y_train (np.ndarray): The output of the PDE.
        snr (float): The signal-to-noise ratio in dB.

    Returns:
        np.ndarray: The noisy output of the PDE.
    """
    sp = np.mean(y_train**2)  # signal power
    snr = 10 ** (snr / 10)  # convert dB to linear scale
    noise_power = sp / snr  # noise power
    std = np.sqrt(noise_power)  # noise standard deviation
    noise = np.random.normal(0, std, size=y_train.shape)  # generate noise
    y_noisy = y_train + noise  # add noise to output
    return y_noisy
