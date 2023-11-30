import numpy as np

def add_noise(y_train, snr):
    sp = np.mean(y_train**2) # signal power
    snr = 10**(snr/10) # convert dB to linear scale
    noise_power = sp / snr # noise power
    std = np.sqrt(noise_power) # noise standard deviation
    noise = np.random.normal(0, std, size=y_train.shape) # generate noise
    y_noisy = y_train + noise # add noise to output
    return y_noisy
