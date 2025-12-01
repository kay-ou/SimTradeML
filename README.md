# SimTradeML

**可复用的量化交易机器学习框架**

SimTradeML 是一个设计简洁、易于扩展的机器学习训练框架，专为量化交易场景设计。无缝集成 [SimTradeLab](https://github.com/kay-ou/SimTradeLab)，直接读取本地 h5 数据文件进行模型训练。

## 特性

- 🎯 **解耦设计** - 数据、特征、模型各层独立
- 🔌 **可插拔** - 支持多种数据源、特征、模型
- 🚀 **开箱即用** - 集成SimTradeLab，复制h5文件即可训练
- 📊 **完整评估** - IC/ICIR/分位收益/方向准确率
- ⚡ **高性能** - 支持多进程并行采样

## 快速开始

### 安装

```bash
# 克隆项目
cd /home/kay/dev/SimTradeML

# 安装依赖（包含SimTradeLab）
poetry install
pip install simtradelab

# 或者使用extras
poetry install -E simtradelab
```

### 准备数据

将 SimTradeLab 的 h5 数据文件复制到 `data/` 目录：

```bash
data/
├── ptrade_data.h5           # 价格数据
└── ptrade_fundamentals.h5   # 基本面数据
```

### 运行MVP训练

```bash
poetry run python examples/mvp_train.py
```

## 项目结构

```
SimTradeML/
├── src/simtrademl/
│   ├── core/                    # 核心框架
│   │   ├── data/               # 数据抽象层
│   │   │   ├── base.py        # DataSource基类
│   │   │   └── collector.py   # 数据收集器
│   │   └── utils/              # 工具模块
│   │       ├── config.py      # 配置管理
│   │       ├── logger.py      # 日志系统
│   │       └── metrics.py     # 评估指标
│   │
│   └── data_sources/           # 数据源实现
│       └── simtradelab_source.py  # SimTradeLab数据源
│
├── examples/
│   ├── mvp_train.py           # MVP训练脚本
│   └── README_MVP.md          # MVP使用说明
│
├── pyproject.toml
└── README.md
```

## MVP版本

当前实现的是最小可用版本（MVP），包含：

✅ **核心功能：**
- SimTradeLab 数据源集成
- 简单技术指标特征（11个）
- XGBoost 模型训练
- 时间序列分割（70/15/15）
- IC/Rank IC/分位收益评估

🚧 **后续扩展：**
- 特征工程框架（Feature注册、自定义特征）
- 特征选择（相关性、IC筛选、VIF）
- 多模型支持（LightGBM、CatBoost）
- 超参数优化（Optuna）
- 完整Pipeline

## 使用示例

### 基础用法（MVP）

```python
from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource

# 1. 初始化数据源
data_source = SimTradeLabDataSource()

# 2. 获取数据
stocks = data_source.get_stock_list()
price_df = data_source.get_price_data('600519.SS')

# 3. 训练模型（见 examples/mvp_train.py）
```

### 配置系统

```python
from simtrademl.core.utils.config import Config

# 从dict创建配置
config = Config.from_dict({
    'data': {
        'lookback_days': 60,
        'predict_days': 5,
    },
    'model': {
        'type': 'xgboost',
        'params': {'max_depth': 4},
    }
})

# 访问配置（支持点号语法）
lookback = config.get('data.lookback_days')  # 60
```

## 评估指标

SimTradeML 提供完整的量化评估指标：

- **IC (Information Coefficient)** - 预测值与实际收益的相关性
- **Rank IC** - 排序相关性（Spearman）
- **ICIR** - IC信息比率（IC均值/IC标准差）
- **分位收益** - 按预测值分组的收益（每日再平衡）
- **方向准确率** - 预测方向正确的比例

## 依赖

核心依赖：
- Python >= 3.9
- numpy, pandas, scipy
- scikit-learn
- xgboost
- pyyaml, tqdm

可选依赖：
- simtradelab - 数据集成
- optuna - 超参数优化
- mlflow - 实验追踪
- tsfresh - 时间序列特征

## 开发计划

### Phase 1: MVP ✅
- [x] 核心配置系统
- [x] 数据层抽象
- [x] SimTradeLab集成
- [x] 基础训练流程
- [x] 评估指标

### Phase 2: 特征工程 🚧
- [ ] Feature基类和注册表
- [ ] 预定义特征库（技术、基本面、市场）
- [ ] 特征选择器（相关性、IC、VIF）
- [ ] 自定义特征支持

### Phase 3: Pipeline 🚧
- [ ] TrainingPipeline
- [ ] ModelEvaluator
- [ ] CrossValidator
- [ ] Model基类

### Phase 4: 扩展功能 ⏳
- [ ] 多模型支持
- [ ] 超参数优化
- [ ] MLflow集成
- [ ] 更多数据源

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

---

**Note:** 当前为MVP版本，专注核心功能验证。根据测试结果决定后续扩展方向。
