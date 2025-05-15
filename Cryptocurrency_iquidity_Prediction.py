#  Import Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

#  Step 1: Load both CSV files
df1 = pd.read_csv("dataset/coin_gecko_2022-03-16.csv")
df2 = pd.read_csv("dataset/coin_gecko_2022-03-17.csv")

# Step 2: Combine the two datasets
df = pd.concat([df1, df2], ignore_index=True)

# Step 3: Drop rows with missing values
df_cleaned = df.dropna()

#  Step 4: Convert 'date' column to datetime
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

#  Step 5: Normalize numerical columns
numerical_cols = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']
scaler = MinMaxScaler()
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

#  Optional: Save cleaned data
df_cleaned.to_csv("dataset/cleaned_crypto_data.csv", index=False)

#  Step 6: Feature Engineering
df_fe = df_cleaned.copy()
df_fe.sort_values(by=['coin', 'date'], inplace=True)

# Rolling features (NaNs retained)
df_fe['price_ma_2'] = df_fe.groupby('coin')['price'].transform(lambda x: x.rolling(window=2).mean())
df_fe['price_std_2'] = df_fe.groupby('coin')['price'].transform(lambda x: x.rolling(window=2).std())

# Custom liquidity-related features
df_fe['liquidity_ratio'] = df_fe['24h_volume'] / (df_fe['price'] + 1e-6)
df_fe['cap_to_volume'] = df_fe['mkt_cap'] / (df_fe['24h_volume'] + 1e-6)
df_fe['weighted_change'] = df_fe['24h'] * df_fe['24h_volume']

#  Save final dataset for modeling
df_fe.to_csv("dataset/feature_engineered_crypto_data.csv", index=False)

#  Preview
print("Final Dataset Preview:")
print(df_fe[['coin', 'date', 'price', 'price_ma_2', 'liquidity_ratio', 'cap_to_volume', 'weighted_change']].head())

# Filter clean rows for modeling
features = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap', 'cap_to_volume', 'weighted_change']
df_model = df_fe.dropna(subset=features + ['liquidity_ratio'])

X = df_model[features]
y = df_model['liquidity_ratio']

# Train on entire dataset (no split due to small size)
model = LinearRegression()
model.fit(X, y)

# Predict + Evaluate
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Model Evaluation:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R^2 Score:", r2)



# Save the trained model
joblib.dump(model, "model/liquidity_predictor.pkl")
