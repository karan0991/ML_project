
# 💸 Cryptocurrency Liquidity Prediction

This project aims to predict the **liquidity ratio** of cryptocurrencies using machine learning models based on historical market data such as price, volume, and volatility indicators.

---

## 📌 Problem Statement

Cryptocurrency markets are volatile, and liquidity plays a vital role in market stability. This project forecasts liquidity to help traders and exchanges detect potential crises and manage risk effectively.

---

## 📊 Features
- Raw features:
- `price`, `1h`, `24h`, `7d` percent changes
- `24h_volume` and `mkt_cap`
- Engineered features:
  - `cap_to_volume` = Market Cap / Volume
  - `weighted_change` = 24h Change × Volume
  - `liquidity_ratio` = Volume / Price

---

## 🧠 Model

- **Linear Regression** trained on engineered features
- Evaluation Metrics:
  - RMSE, MAE, R² (on training data)

---

## 📈 EDA

Visualizations include:
- Price distribution
- Liquidity ratio vs volume
- Feature correlation heatmap

---

## 🧪 Deployment

Built with **Streamlit** for interactive predictions:
```bash
streamlit run app.py
```

---

## 📂 File Structure

```
.
├── app.py
├── models/
│   └── liquidity_predictor.pkl
├── dataset/
│   ├── coin_gecko_2022-03-16.csv
│   ├── coin_gecko_2022-03-17.csv
│   ├── cleaned_crypto_data.csv
│   └── feature_engineered_crypto_data.csv
├── Crypto_Liquidity_Prediction_Report.pdf
└── README.md


```

---

## ✍️ Author

Karan Patel  
Machine Learning Project – Cryptocurrency Liquidity Prediction
