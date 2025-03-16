from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from flask_cors import CORS
from sklearn.metrics import r2_score


app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    return model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker, start_date, end_date, future_days = data['ticker'], data['start_date'], data['end_date'], int(data['future_days'])

    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    X, y, scaler = preprocess_data(stock_data)

    # Train CNN model
    cnn_model = build_cnn_model((X.shape[1], X.shape[2]))
    cnn_model.compile(optimizer='adam', loss='mse')
    cnn_model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Train SVM model
    cnn_features = cnn_model.predict(X)
    svm = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svm.fit(cnn_features, y)

    # Predict Future Prices (Rolling Forecast)
    future_prices = []
    last_60_days = scaler.transform(stock_data.values[-60:].reshape(-1, 1))  # Take last 60 days and scale

    for i in range(future_days):
        future_X = np.array([last_60_days[:, 0]])
        future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))

        cnn_feature = cnn_model.predict(future_X)
        next_price_scaled = svm.predict(cnn_feature)
        next_price = scaler.inverse_transform(next_price_scaled.reshape(-1, 1))[0][0]

        future_prices.append({
            'date': (datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d'),
            'price': round(next_price, 2)
        })

        last_60_days = np.append(last_60_days[1:], next_price_scaled).reshape(-1, 1)
        # Predict on the test set
    cnn_features_test = cnn_model.predict(X)
    svm_predictions = svm.predict(cnn_features_test)
    svm_predictions_rescaled = scaler.inverse_transform(svm_predictions.reshape(-1, 1))

# Rescale actual values
    y_actual_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

# Compute R² score
    r2 = r2_score(y_actual_rescaled, svm_predictions_rescaled)

# Convert R² score to a percentage (optional)
    accuracy = max(0, r2 * 100)


    #accuracy = np.random.uniform(80, 95)  # Mock accuracy calculation

    return jsonify({"accuracy": accuracy, "future_prices": future_prices})

if __name__ == '__main__':
    app.run(debug=True)
