# SimTradeML MVP - Quick Start

最小可用版本，验证核心流程。

## 快速开始

### 1. 安装依赖

```bash
cd /home/kay/dev/SimTradeML
poetry install
pip install simtradelab  # 安装 SimTradeLab
```

### 2. 准备数据

将 SimTradeData 的 parquet 数据复制到 data 目录：

```bash
# 确保 data 目录包含 parquet 数据
cp -r /path/to/SimTradeData/data/* data/
```

### 3. 运行MVP训练

```bash
poetry run python examples/mvp_train.py
```

## MVP功能

- ✅ 使用 SimTradeLab Research API 读取本地 parquet 数据
- ✅ 计算简单技术指标特征（MA、收益率、波动率等）
- ✅ 时间序列分割（70% 训练，15% 验证，15% 测试）
- ✅ XGBoost 模型训练
- ✅ IC/Rank IC/分位收益评估
- ✅ 模型保存

## 输出文件

- `examples/mvp_train.log` - 训练日志
- `examples/mvp_model.json` - 训练好的模型

## 下一步

MVP验证通过后，可以扩展：

1. 添加更多特征（基本面、市场情绪等）
2. 特征选择（相关性过滤、IC筛选）
3. 超参数优化（Optuna）
4. 多模型支持（LightGBM、CatBoost）
5. 完整的Pipeline框架

## 当前限制

- 只使用前100只股票（测试用）
- 每10天采样一次（加快速度）
- 简单特征（11个技术指标）
- 单一模型（XGBoost）

根据MVP测试结果决定是否继续扩展框架。
