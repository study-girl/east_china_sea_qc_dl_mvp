import pandas as pd
import numpy as np

def clean_data(df):
    df_clean = df.copy()
    # 缺失值填充
    df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)
    return df_clean
