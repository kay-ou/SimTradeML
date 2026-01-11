# SimTradeML

量化交易机器学习框架，专为 A 股设计，无缝集成 [SimTradeLab](https://github.com/kay-ou/SimTradeLab)。

## 快速开始

### 安装

```bash
cd /path/to/SimTradeML
poetry install
pip install simtradelab  # 如果需要使用 SimTradeLab 数据源
```

### 5分钟训练第一个模型

```bash
# 1. 准备数据（复制 SimTradeLab 的 h5 文件到 data/ 目录）
mkdir -p data
cp /path/to/ptrade_data.h5 data/
cp /path/to/ptrade_fundamentals.h5 data/

# 2. 运行训练
poetry run python examples/mvp_train.py
```

### 基础用法

```python
from simtrademl import Config, setup_logger
from simtrademl.data_sources import SimTradeLabDataSource
from simtrademl.core.data.collector import DataCollector

# 1. 配置
config = Config.from_dict({
    'data': {'lookback_days': 60, 'predict_days': 5},
    'training': {'parallel_jobs': 4}
})

# 2. 数据源
data_source = SimTradeLabDataSource()  # 自动读取 data/ 目录

# 3. 收集训练数据
collector = DataCollector(data_source, config)
X, y, dates = collector.collect()

# 4. 训练模型（参考 examples/mvp_train.py）
```

## 核心特性

- **数据源抽象**: 轻松切换不同数据源
- **特征工程**: 内置技术指标，支持自定义
- **评估指标**: IC/ICIR/分位收益/方向准确率
- **并行处理**: 自动多进程采样加速

## 项目结构

```
src/simtrademl/
├── core/
│   ├── data/          # 数据层（DataSource, DataCollector）
│   └── utils/         # 工具（Config, Logger, Metrics）
└── data_sources/      # 数据源实现
    └── simtradelab_source.py

examples/
└── mvp_train.py       # 完整训练示例
```

## API 文档

### 配置管理

```python
from simtrademl import Config

# 从字典创建
config = Config.from_dict({'data': {'lookback_days': 60}})

# 从 YAML 加载
config = Config.from_yaml('config.yml')

# 点号访问
lookback = config.get('data.lookback_days', default=30)
config.set('model.type', 'xgboost')
```

### 数据收集

```python
from simtrademl.core.data.collector import DataCollector

collector = DataCollector(data_source, config)

# 收集所有股票
X, y, dates = collector.collect()

# 过滤股票
X, y, dates = collector.collect(
    stock_filter=lambda s: s.startswith('60')
)

# 自定义特征
def custom_features(stock, price_df, idx, date, ds):
    return {'my_feature': price_df['close'].iloc[idx-1]}

collector = DataCollector(data_source, config,
                          feature_calculator=custom_features)
```

### 评估指标

```python
from simtrademl import (
    calculate_ic, calculate_rank_ic, calculate_icir,
    calculate_quantile_returns, calculate_direction_accuracy
)

# IC 指标
ic, p_value = calculate_ic(predictions, actuals)
rank_ic, p_value = calculate_rank_ic(predictions, actuals)
icir, ic_std = calculate_icir(predictions, actuals)

# 分位收益（日度再平衡）
quantile_returns, long_short = calculate_quantile_returns(
    predictions, actuals, dates=sample_dates
)

# 方向准确率
accuracy = calculate_direction_accuracy(predictions, actuals)
```

## 测试

```bash
# 运行所有测试
poetry run pytest

# 查看覆盖率
poetry run pytest --cov=simtrademl --cov-report=html
open htmlcov/index.html
```

## 配置示例

完整配置（`config.yml`）:

```yaml
data:
  lookback_days: 60
  predict_days: 5
  sampling_window_days: 15

model:
  type: xgboost
  params:
    max_depth: 4
    learning_rate: 0.04
    subsample: 0.7
    colsample_bytree: 0.7

training:
  train_ratio: 0.70
  val_ratio: 0.15
  parallel_jobs: -1  # -1 = 使用所有 CPU
```

## 扩展数据源

```python
from simtrademl.core.data.base import DataSource

class MyDataSource(DataSource):
    def get_stock_list(self) -> List[str]:
        return ['600519.SS', '000858.SZ']

    def get_price_data(self, stock, start_date, end_date, fields):
        # 返回 DataFrame，index 为日期
        return pd.DataFrame({
            'open': [...], 'high': [...], 'low': [...],
            'close': [...], 'volume': [...]
        })

    # 实现其他必需方法...
```

## 依赖

**核心**: Python 3.9+, numpy, pandas, scikit-learn, xgboost 0.90
**可选**: simtradelab (数据), optuna (超参优化), mlflow (实验追踪)

## 开发计划

- [x] MVP: 数据收集 + XGBoost 训练
- [ ] 特征工程框架
- [ ] 多模型支持（LightGBM, CatBoost）
- [ ] 超参数优化集成

## 许可证

MIT License

---

**文档**: 参考 `examples/mvp_train.py` 获取完整示例
**问题**: 提交 Issue 到 GitHub
**测试覆盖率**: 88% | 66 个测试全部通过
