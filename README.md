# 东海海域台风观测数据深度学习质控 (Deep Learning Sea QC MVP)

## 项目简介
本项目针对东海海域台风影响下的浮标与船舶观测数据，构建了一个 **深度学习驱动的多参数质控系统**。  
通过 LSTM Autoencoder 对风速、风向、波高、海温、盐度、流速、流向等数据进行异常检测，并结合多浮标数据融合实现异常修正与可视化报告生成。

---

## 功能特点
- **数据清洗**：填充缺失值，保证数据完整性  
- **深度学习异常检测**：使用 LSTM Autoencoder 自动标记异常观测  
- **异常修正 & 多浮标融合**：利用邻近浮标和滑动窗口均值修正异常值  
- **报告生成 & 可视化**：输出质控报告和异常点时间序列图  
- **可扩展**：可加入更多浮标数据、调整参数或替换模型  

---

## 文件结构
```text
east_china_sea_qc_dl_mvp/
├── data/ # 示例观测数据
│   └── sample_data.csv
├── agents/ # 四个功能 Agent
│   ├── data_cleaning.py
│   ├── anomaly_detection.py
│   ├── data_fusion_correction.py
│   └── report_visualization.py
├── main.py # 主程序
├── README.md # 项目说明
└── .gitignore # Git 忽略文件
```

---

## 技术栈
- Python 3.9+  
- 数据处理：`pandas`, `numpy`  
- 深度学习：`tensorflow` (LSTM Autoencoder)  
- 可视化：`matplotlib`  
- 可选：`scikit-learn` 用于标准化或评估  

---

## 使用方法
### 安装依赖
```bash
pip install pandas numpy matplotlib tensorflow scikit-learn
```
运行项目
```
python main.py
```
输出
控制台打印：质控报告，包括总记录数、检测到的异常数
CSV 文件：data/qc_corrected_data.csv（质控后的观测数据）
可视化图：data/anomaly_plot.png（时间序列异常点可视化）
使用示例

运行完成后控制台输出：
```
=== 质控报告 ===
total_records: 1000
dl_anomalies_detected: 45
```
同时生成 qc_corrected_data.csv 和 anomaly_plot.png，方便直接用于后续分析或模型输入。

---

## 扩展与改进
替换 LSTM Autoencoder 为 GRU 或 Transformer 模型，提高异常检测精度
添加更多观测参数，如溶解氧、浊度等
支持批量处理多浮标数据文件
可集成 Docker 容器，实现一键运行
参考
LSTM Autoencoder 时间序列异常检测原理
台风影响下海洋浮标观测数据质量控制研究

---
