import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")

# Main title and description
st.title("🔮 Cryptocurrency Liquidity Predictor")
st.markdown("""
Welcome to the **Crypto Liquidity Predictor**!  
Input key market features and get an instant prediction of a cryptocurrency's liquidity ratio — a critical indicator for market stability.
""")

# Sidebar input section
st.sidebar.header("🧮 Input Market Conditions")
price = st.sidebar.slider("💰 Price (normalized)", 0.0, 1.0, 0.5)
change_1h = st.sidebar.slider("⏱️ 1h % Change", 0.0, 1.0, 0.5)
change_24h = st.sidebar.slider("📊 24h % Change", 0.0, 1.0, 0.5)
change_7d = st.sidebar.slider("📅 7d % Change", 0.0, 1.0, 0.5)
volume = st.sidebar.slider("🔄 24h Volume (normalized)", 0.0, 1.0, 0.5)
market_cap = st.sidebar.slider("🏦 Market Cap (normalized)", 0.0, 1.0, 0.5)

# Feature engineering
cap_to_volume = market_cap / (volume + 1e-6)
weighted_change = change_24h * volume

# Feature array
features = np.array([[price, change_1h, change_24h, change_7d,
                      volume, market_cap, cap_to_volume, weighted_change]])

# Load the trained model
model = joblib.load("model/liquidity_predictor.pkl")

# Predict
prediction = model.predict(features)[0]

# Display result
st.markdown("---")
st.subheader("📈 Predicted Liquidity Ratio")
st.metric(label="Liquidity Ratio", value=f"{round(prediction, 4)}")

# Footer
st.markdown("---")
st.markdown("""
Developed with ❤️ using Streamlit  
🔗 [GitHub Repository](https://github.com/karan0991/ML_project.git) • 🧠 Powered by Machine Learning  
""")
