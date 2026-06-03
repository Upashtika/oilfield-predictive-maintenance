from scipy.io import loadmat
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd
import os

data_rows = []

datasets = [
    ("data/bearing/normal", "Normal"),
    ("data/bearing/inner_race_fault", "Inner"),
    ("data/bearing/outer_race_fault", "Outer"),
    ("data/bearing/ball_fault", "Ball")
]

window_size = 5000

for folder, label in datasets:

    for file in os.listdir(folder):

        if not file.endswith(".mat"):
            continue

        filepath = os.path.join(folder, file)

        data = loadmat(filepath)

        signal_key = None

        for key in data.keys():
            if "DE_time" in key:
                signal_key = key
                break

        if signal_key is None:
            continue

        signal = data[signal_key].flatten()

        for start in range(0, len(signal) - window_size, window_size):

            window = signal[start:start + window_size]

            rms = np.sqrt(np.mean(window**2))
            peak = np.max(np.abs(window))
            kurt = kurtosis(window)
            skw = skew(window)

            data_rows.append([
                rms,
                peak,
                kurt,
                skw,
                label
            ])

df = pd.DataFrame(
    data_rows,
    columns=[
        "RMS",
        "Peak",
        "Kurtosis",
        "Skewness",
        "Fault"
    ]
)

df.to_csv(
    "data/bearing/bearing_features.csv",
    index=False
)

print("Dataset Created Successfully")
print("Shape:", df.shape)
print(df.head())