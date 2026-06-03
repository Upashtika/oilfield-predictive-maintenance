from scipy.io import loadmat
from scipy.stats import kurtosis, skew
import numpy as np

data = loadmat(
    "data/bearing/normal/Normal_0.mat"
)

signal = data["X097_DE_time"]
print("Shape:", signal.shape)
print(signal[:10])

signal =signal.flatten()
rms = np.sqrt(np.mean(signal**2))
peak = np.max(np.max(np.abs(signal)))
kurt = kurtosis(signal)
skw = skew(signal)

print("RMS:", rms)
print("Peak:", peak)
print("Kurtosis:", kurt)
print("Skewness:", skw)