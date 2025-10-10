# 📦 SimTradeML

**SimTradeML** is a modular, cloud-ready framework for training, evaluating, and deploying machine learning models in financial simulation environments. It serves as the predictive engine behind the SimTrade ecosystem, bridging raw data from [SimTradeData](https://github.com/ykayz/SimTradeData) and strategy logic in [SimTradeLab](https://github.com/ykayz/SimTradeLab).

---

## 🎯 Purpose

SimTradeML provides reusable ML pipelines and model services tailored for financial workflows. It enables:

- Time series forecasting, classification, and regression for market signals
- Model packaging as RESTful APIs for strategy integration
- Cloud-native deployment via AWS CDK and GitLab CI
- Feedback loops between strategy performance and model retraining

---

## 🧩 Architecture Overview

```
simtradeML/
├── forecast_engine/      # ML models: ARIMA, Prophet, LSTM, XGBoost, etc.
├── model_api/            # FastAPI endpoints for model inference
├── data_ingest/          # ETL pipelines consuming SimTradeData outputs
├── infra/                # AWS CDK / Terraform templates for deployment
├── ci_pipeline/          # GitLab CI/CD scripts
├── notebooks/            # Exploratory analysis and prototyping
├── tests/                # Unit and integration tests
└── README.md
```

---

## 🔗 Ecosystem Integration

SimTradeML is designed to work seamlessly with other SimTrade modules:

| Module         | Role                                  | Integration Method |
|----------------|----------------------------------------|---------------------|
| SimTradeData   | Provides structured financial data     | API / local DB      |
| SimTradeLab    | Consumes model outputs for strategies  | `.pkl` / `.h5` files or API |
| Ptrade         | Embeds trained models in strategy code | Upload to research tab or call via script |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ykayz/SimTradeML.git
cd SimTradeML
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train a sample model

```bash
python forecast_engine/train_xgboost.py
```

### 4. Launch model API

```bash
uvicorn model_api.main:app --reload
```

---

## 🧠 Example Use Case

Train a volatility prediction model using ETF data from SimTradeData, deploy it as an API, and call it from a SimTradeLab strategy to generate dynamic position sizing signals.

---

## 🛠️ Technologies Used

- Python, scikit-learn, XGBoost, statsmodels
- FastAPI, Docker, AWS CDK
- GitLab CI/CD
- Streamlit (optional dashboard)
- Jupyter Notebooks for prototyping

---

## 📌 Status

Actively evolving. Initial modules include:

- ETF volatility prediction
- Trend classification
- Strategy-model feedback loop prototype

---

## 🤝 Contributing

Contributions are welcome! Please submit issues or pull requests for:

- New model types
- Deployment templates
- Strategy integration examples
- Documentation improvements

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
```
