# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import nltk
import openai

# Streamlit UI setup
st.set_page_config(page_title="Stock Analysis and Prediction", layout="wide")
st.title("📈 Real-time Stock Analysis and Prediction Using LSTM & AI Insights")

# Initialize OpenAI GPT API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize VADER sentiment analyzer
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOG):", "AAPL").upper()

# User input for prediction days
future_days = st.slider("Select number of days to predict:", min_value=1, max_value=30, value=7)

# Sidebar header
st.sidebar.subheader("Fetching and Analyzing Data...")

if ticker:
    # Fetch real-time stock data
    stock_data = yf.download(ticker, period="2y")
    
    if stock_data.empty:
        st.sidebar.error("Invalid Ticker! Please enter a valid stock symbol.")
    else:
        st.sidebar.success(f"Data for {ticker} loaded successfully!")
        
        # Prepare Data for LSTM Prediction
        stock_data = stock_data[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])

        def create_sequences(data, time_step=50):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 50
        dataset = stock_data['Scaled_Close'].values.reshape(-1, 1)
        X, Y = create_sequences(dataset, time_step)

        train_size = int(len(X) * 0.8)
        X_train, Y_train = X[:train_size], Y[:train_size]
        X_test, Y_test = X[train_size:], Y[train_size:]

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build and Train LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        with st.spinner("Training LSTM Model... ⏳"):
            model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0)

        # Generate Predictions
        Y_pred = model.predict(X_test)
        Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # Calculate Model Metrics
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        # Display Accuracy Metrics
        st.subheader("📊 Model Accuracy Metrics")
        st.write(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"🔹 R² Score: {r2:.4f}")

        # Predict Future Stock Prices
        def predict_future_prices(model, last_50_days, future_days):
            future_predictions = []
            current_input = last_50_days.reshape(1, -1, 1)
            for _ in range(future_days):
                next_prediction = model.predict(current_input)[0][0]
                future_predictions.append(next_prediction)
                current_input = np.append(current_input[:, 1:, :], [[[next_prediction]]], axis=1)
            return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        last_50_days = dataset[-time_step:]
        future_prices = predict_future_prices(model, last_50_days, future_days)
        future_dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices.flatten()})
        st.subheader(f"📅 Predicted Stock Prices for Next {future_days} Days")
        st.dataframe(future_df)

        # Plot Actual vs Predicted Prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index[-len(Y_test):], y=Y_test.flatten(),
                                 mode='lines', name='Actual Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=stock_data.index[-len(Y_test):], y=Y_pred.flatten(),
                                 mode='lines', name='Predicted Price', line=dict(color='red', dash='dot')))
        fig.update_layout(title=f"Actual vs Predicted Prices ({ticker})",
                          xaxis_title="Date", yaxis_title="Stock Price (USD)")
        st.plotly_chart(fig)

        # Plot Future Predictions
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'],
                                        mode='lines+markers', name='Future Predictions', line=dict(color='green')))
        fig_future.update_layout(title=f"Future Stock Price Predictions ({ticker})",
                                 xaxis_title="Date", yaxis_title="Predicted Price (USD)")
        st.plotly_chart(fig_future)

        # Sentiment Analysis with OpenAI GPT
        def get_gpt_insights(news_text):
            prompt = f"Summarize the following stock news and provide key takeaways:\n\n{news_text}"
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=150
                )
                return response.choices[0].text.strip()
            except Exception as e:
                return f"Error fetching insights from OpenAI GPT: {e}"

        def fetch_stock_news(stock_name):
            try:
                news_articles = yf.Ticker(stock_name).news
                return "\n".join([article['title'] for article in news_articles[:5]])
            except Exception:
                return "No news found

 
