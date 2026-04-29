import matplotlib.pyplot as plt

def generate_report(df):
    report = {
        'total_records': len(df),
        'dl_anomalies_detected': df['dl_anomaly'].sum()
    }
    # 可视化
    plt.figure(figsize=(10,6))
    plt.plot(df['timestamp'], df['wind_speed'], label='Wind Speed')
    plt.scatter(df['timestamp'][df['dl_anomaly']], df['wind_speed'][df['dl_anomaly']], color='r', label='Anomaly')
    plt.xlabel('Time')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed & DL Detected Anomalies')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/anomaly_plot.png')
    plt.close()
    return report
