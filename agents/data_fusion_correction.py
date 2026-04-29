import pandas as pd
import numpy as np

def correct_anomalies(df):
    df_corr = df.copy()
    for col in ['wind_speed','wave_height','sea_temp','salinity','current_speed']:
        anomaly_idx = df_corr[df_corr['dl_anomaly']].index
        for idx in anomaly_idx:
            neighbors = df_corr.loc[max(idx-1,0):min(idx+1,len(df_corr)-1), col]
            df_corr.at[idx, col] = neighbors[~neighbors.index.isin([idx])].mean()
    return df_corr
