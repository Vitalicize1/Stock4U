# agents/memory.py
import pandas as pd
import os

def save_memory(row, filename="memory.csv"):
    df = pd.DataFrame([row])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
