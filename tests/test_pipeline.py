import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_processing import process_raw_data

df = pd.read_csv('data/raw/data.csv')
print(f"Raw data: {df.shape}")

features = process_raw_data(df)
print(f"\nProcessed features: {features.shape}")
print(features.head())