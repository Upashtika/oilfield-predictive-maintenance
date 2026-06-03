import time
import random
import pandas as pd

while True:
    telemetry = {
        "Air temperature [k]": round(random.uniform(295,305), 2),
        "Process temperature [k]": round(random.uniform(305,315), 2),
        "Rotational speed [rpm]": random.randint(1200, 1600),
        "Torque [Nm]": round(random.uniform(20, 80), 2),
        "Tool wear [min]": random.randint(0, 250)
    }
    df = pd.DataFrame([telemetry])
    print("\nLive Telemetry:")
    print(df)
    time.sleep(2)