import pandas as pd
from agents.data_cleaning import clean_data
from agents.anomaly_detection import detect_anomalies
from agents.data_fusion_correction import correct_anomalies
from agents.report_visualization import generate_report

# 1. 读取数据
data = pd.read_csv('data/sample_data.csv', parse_dates=['timestamp'])

# 2. 数据清洗 Agent
data_cleaned = clean_data(data)

# 3. 异常检测 Agent (深度学习)
data_anomaly = detect_anomalies(data_cleaned, epochs=5)

# 4. 多浮标融合 & 异常修正 Agent
data_corrected = correct_anomalies(data_anomaly)

# 5. 生成报告 & 可视化 Agent
report = generate_report(data_corrected)

# 6. 输出
print('=== 质控报告 ===')
for k,v in report.items():
    print(f'{k}: {v}')

data_corrected.to_csv('data/qc_corrected_data.csv', index=False)
print('\n质控后数据已保存到 data/qc_corrected_data.csv')
print('异常可视化图已保存到 data/anomaly_plot.png')
