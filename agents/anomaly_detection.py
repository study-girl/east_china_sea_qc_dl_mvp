import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler

def detect_anomalies(df, sequence_length=3, epochs=5):
    df_numeric = df[['wind_speed','wave_height','sea_temp','salinity','current_speed']]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_numeric)
    
    # 构建滑动窗口
    sequences = []
    for i in range(len(data_scaled)-sequence_length):
        sequences.append(data_scaled[i:i+sequence_length])
    sequences = np.array(sequences)
    
    # LSTM Autoencoder
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(sequence_length, data_scaled.shape[1])),
        RepeatVector(sequence_length),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(data_scaled.shape[1]))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(sequences, sequences, epochs=epochs, batch_size=8, verbose=0)
    
    # 重建误差
    reconstructed = model.predict(sequences)
    mse = np.mean(np.square(reconstructed - sequences), axis=(1,2))
    threshold = np.percentile(mse, 95)  # 异常阈值
    anomaly_flags = np.zeros(len(df_numeric), dtype=bool)
    anomaly_flags[sequence_length:len(df_numeric)] = mse > threshold
    df['dl_anomaly'] = anomaly_flags
    return df
