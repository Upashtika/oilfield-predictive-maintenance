import pandas as pd
df = pd.read_csv("data/ai4i/ai4i2020.csv")
print(df.head(5))
print("\nShape")
print(df.shape)
print("\nColumns")
print(df.columns)
