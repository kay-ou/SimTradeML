# 📦 SimTradeML

**SimTradeML** 是 SimTrade 生态中的智能预测引擎,专注于将金融数据转化为可直接使用的机器学习模型。它为量化研究者提供了一个轻量级的模型训练平台,输出与 Ptrade 框架完全兼容的 `.pkl` 模型文件,可无缝集成到 [SimTradeLab](https://github.com/kay_ou/SimTradeLab) 和 Ptrade 的策略代码中。

---

## 🎯 核心定位

**简单直接的模型训练工具** — 专注于训练,不做部署

- **数据输入**: 读取 [SimTradeData](https://github.com/kay_ou/SimTradeData) 提供的 `.h5` 格式金融数据
- **模型训练**: 提供标准化的训练、评估和超参数优化流程
- **模型输出**: 导出兼容 Ptrade 框架的 `.pkl` 模型文件
- **策略集成**: 训练好的模型可直接被 SimTradeLab/Ptrade 策略加载调用

---

## 🧩 核心功能

- 时间序列预测、分类、回归模型训练
- 基于 Optuna 的自动化超参数优化
- 模型持久化为 Ptrade 兼容的 `.pkl` 格式
---

## 🔗 生态协作

SimTradeML 在 SimTrade 生态中的位置:

| 模块           | 职责                     | 与 SimTradeML 的关系               |
|---------------|--------------------------|-----------------------------------|
| SimTradeData  | 提供标准化金融数据         | SimTradeML 读取其导出的 `.h5` 数据文件 |
| SimTradeLab   | 策略回测和研究平台         | 加载 SimTradeML 训练的 `.pkl` 模型   |
| Ptrade        | 实盘/模拟交易执行         | 加载 SimTradeML 训练的 `.pkl` 模型   |

**数据流**: SimTradeData (`.h5`) → SimTradeML (训练) → `.pkl` 模型 → SimTradeLab/Ptrade (策略)

---

## 🧠 典型工作流

1. **准备数据**: 使用 SimTradeData 获取股票历史数据并导出为 `.h5` 文件
2. **训练模型**: 在 SimTradeML 中训练波动率预测模型,自动优化超参数
3. **导出模型**: 将训练好的模型保存为 Ptrade 兼容的 `.pkl` 文件
4. **策略集成**: 在 SimTradeLab/Ptrade 策略中加载模型进行预测
5. **迭代优化**: 根据回测结果调整特征或参数,重新训练模型

---

## 🛠️ 技术栈

SimTradeML 使用的机器学习库均兼容 Ptrade 环境:

- **机器学习框架** (会被序列化到模型):
  - scikit-learn 0.18 (分类、回归、聚类)
  - XGBoost 0.6a2 (梯度提升树)
  - statsmodels 0.10.2 (时间序列、统计模型)

- **深度学习** (可选):
  - TensorFlow 1.3.0rc1
  - Keras 2.2.4
  - Theano 0.8.2

- **数据处理**:
  - NumPy 1.11.2
  - pandas 0.23.4
  - h5py 2.6.0 (读取 SimTradeData 数据)

- **训练辅助工具** (仅训练阶段使用,不影响模型兼容性):
  - Optuna (超参数优化)
  - MLflow (实验追踪和模型版本管理)
  - SHAP (模型解释和特征重要性分析)
  - imbalanced-learn (处理不平衡数据集)
  - tsfresh (时间序列自动特征提取)
  - pandera (数据验证)
  - TA-Lib 0.4.10 (技术指标计算)
  - hmmlearn 0.2.0 (隐马尔可夫模型)
  - PyBrain 0.3 (神经网络)

**开发环境**: Poetry, pytest, Jupyter Notebook

> 注: 标注版本号的库需与 Ptrade 环境保持一致,确保训练的模型可直接在 Ptrade 中加载使用。训练辅助工具不会被序列化到模型文件中,可自由选择版本

---

## 📌 项目状态

持续开发中,已完成:

- ✅ 模型训练与评估框架
- ✅ Optuna 超参数优化
- ✅ Ptrade 兼容的模型持久化接口

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request:

- 新的模型类型和特征工程方法
- 与 SimTradeLab/Ptrade 的集成示例
- 训练脚本和最佳实践
- 文档完善

---

## 📄 许可证

MIT License. 详见 [LICENSE](LICENSE)。
