import numpy as np


def augment_by_noise(y_train: np.ndarray, snr: float, seed: int) -> np.ndarray:
    """Adds noise to the output of the PDE.

    Args:
        y_train (np.ndarray): The output of the PDE.
        snr (float): The signal-to-noise ratio in dB.

    Returns:
        np.ndarray: The noisy output of the PDE.
    """
    # create default_rng object with seed
    rng = np.random.default_rng(seed)

    # create noise with same shape as y_train
    noise_power = np.divide(np.mean(y_train**2), 10 ** (snr / 10))
    std = np.sqrt(noise_power)
    noise = rng.normal(0, std, size=y_train.shape)

    # augment y_train with noise
    return y_train + noise
