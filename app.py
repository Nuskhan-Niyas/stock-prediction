

import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Fetch Stock Data Function
def fetch_stock_data(stock_symbol, start="2012-01-01"):
    """Fetch historical stock price data from Yahoo Finance"""
    end = datetime.datetime.today().strftime('%Y-%m-%d')
    data = yf.download(stock_symbol, start, end)
    data.dropna(inplace=True)
    return data

# ✅ Load Fine-Tuned Models
bi_gru_model = load_model("bi_gru_tuned_stock_1.keras")
cnn_model = load_model("cnn_tuned_stock.keras")
cnn_bi_gru_model = load_model("cnn_bi_gru_tuned_stock.keras")

# ✅ Initialize Streamlit App
st.title("📈 Stock Price Prediction with Bi-GRU, CNN, and CNN + Bi-GRU")

# ✅ User Input for Stock Symbol
stock = st.text_input("Enter Stock Symbol", "TSLA")

# ✅ Fetch & Preprocess Data
stock_data = fetch_stock_data(stock)

# ✅ Properly Fit MinMaxScaler to Ensure Consistent Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close']])  # Ensure consistent scaling

# ✅ Prepare Data for Prediction
X_test = []
for i in range(100, len(scaled_data)):
    X_test.append(scaled_data[i-100:i])
X_test = np.array(X_test).reshape(-1, 100, 1)  # Ensure proper shape

# ✅ Prediction & Evaluation Function
def predict_and_evaluate(model, model_name):
    """Predict stock prices and evaluate the model"""
    scale_factor = 1 / scaler.scale_[0]
    Y_pred = model.predict(X_test) * scale_factor
    Y_test_rescaled = stock_data['Close'][100:].values  # Get actual stock prices

    # Compute evaluation metrics
    mae = mean_absolute_error(Y_test_rescaled, Y_pred)
    mse = mean_squared_error(Y_test_rescaled, Y_pred)
    r2 = r2_score(Y_test_rescaled, Y_pred)

    # ✅ Display Model Evaluation Metrics
    st.subheader(f"📊 {model_name} Model Evaluation")
    st.write(f"➡️ Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"➡️ Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"➡️ R² Score: {r2:.4f}")

    # ✅ Display Actual vs. Predicted Prices
    st.subheader(f"📊 {model_name} - Actual vs. Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data.index[100:], Y_pred, 'r', label="Predicted Price")
    ax.plot(stock_data.index[100:], stock_data['Close'][100:], 'g', label="Actual Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    # ✅ Predict Next 14 Days
    future_prices = []
    last_100_days = scaled_data[-100:].reshape(1, 100, 1)
    for _ in range(14):
        future_pred = model.predict(last_100_days)
        future_pred_rescaled = future_pred[0, 0] * scale_factor
        future_prices.append(future_pred_rescaled)

        # ✅ Use Predicted Value as Next Input
        future_pred_scaled = scaler.transform(np.array([[future_pred_rescaled]]))
        future_pred_scaled = future_pred_scaled.reshape(1, 1, 1)
        last_100_days = np.append(last_100_days[:, 1:, :], future_pred_scaled, axis=1)

    # ✅ Show Future Prices in a Table
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=14)
    st.subheader(f"📅 {model_name} - Future Stock Price Predictions (Next 14 Days)")
    future_data = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices})
    st.write(future_data)

# ✅ Run Predictions and Display for Each Model
predict_and_evaluate(bi_gru_model, "Bi-GRU")
predict_and_evaluate(cnn_model, "CNN")
predict_and_evaluate(cnn_bi_gru_model, "CNN + Bi-GRU")





