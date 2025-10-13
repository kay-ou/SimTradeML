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

---

## 🔗 Ecosystem Integration

SimTradeML is designed to work seamlessly with other SimTrade modules:

| Module         | Role                                  | Integration Method |
|----------------|----------------------------------------|---------------------|
| SimTradeData   | Provides structured financial data     | API / local DB      |
| SimTradeLab    | Consumes model outputs for strategies  | `.pkl` / `.h5` files or API |
| Ptrade         | Embeds trained models in strategy code | Upload to research tab or call via script |


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
