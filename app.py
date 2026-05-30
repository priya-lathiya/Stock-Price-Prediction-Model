from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime as dt
from datetime import timedelta
import yfinance as yf
import os

app = Flask(__name__)
model = load_model('stock_dl_model.h5')  # Make sure it's trained on 30→1 or 30→7 logic

@app.route('/', methods=['GET'])
def home():
    from datetime import datetime, timedelta
    today = datetime.today()
    delta = timedelta(days=365)
    return render_template('home.html', today=today, delta=delta)

@app.route('/results', methods=['POST'])
def results():
    stock = request.form.get('stock', 'GOOG')
    start = request.form['start']
    end = request.form['end']

    start_date = dt.datetime.strptime(start, "%Y-%m-%d")
    end_date = dt.datetime.strptime(end, "%Y-%m-%d")

    df = yf.download(stock, start=start_date, end=end_date)
    # ➕ Add Exponential Moving Averages (EMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()

    data_desc = df.describe()

    # Prepare data
    data = df[['Close']]
    training_data_len = int(len(data) * 0.7)
    data_training = data[:training_data_len]
    data_testing = data[training_data_len:]

    scaler = MinMaxScaler()
    scaler.fit(data_training)
    data_training_array = scaler.transform(data_training)

    # Prepare input for test set
    past_30_days = data_training.tail(30)
    final_df = pd.concat([past_30_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test, y_test = [], []
    for i in range(30, input_data.shape[0]):
        x_test.append(input_data[i - 30:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Predict on test set
    y_predicted = model.predict(x_test)
    y_predicted = scaler.inverse_transform(y_predicted)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # EMAs: 20-day and 30-day
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Closing Price', color='gray', linewidth=1)
    plt.plot(df.index, df['EMA_20'], label='20-Day EMA', color='blue', linestyle='--', linewidth=1.2)
    plt.plot(df.index, df['EMA_30'], label='30-Day EMA', color='orange', linestyle='--', linewidth=1.2)

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{stock} Closing Price with 20 & 30-Day Exponential Moving Averages")
    plt.legend()
    plt.grid(True)

    ema_chart_path = "static/ema_20_30.png"
    plt.savefig(ema_chart_path)
    plt.close()


    # Forecast next 7 days using last 30 days
    future_input = input_data[-30:].reshape(1, -1)[0].tolist()
    next_7_days = []
    for i in range(7):
        x_future = np.array(future_input[-30:]).reshape(1, 30, 1)
        pred = model.predict(x_future, verbose=0)[0][0]
        next_7_days.append(pred)
        future_input.append(pred)

    next_7_days = scaler.inverse_transform(np.array(next_7_days).reshape(-1, 1)).flatten()

    # Create dates
    test_dates = df.index[-len(y_test):]
    last_date = df.index[-1]
    next_7_dates = [last_date + timedelta(days=i + 1) for i in range(7)]

    # 📊 Plot combined prediction and forecast
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test, label='Actual Price', linewidth=1)
    plt.plot(test_dates, y_predicted, label='Predicted Price', linewidth=1)
    plt.plot(next_7_dates, next_7_days, label='Next 7 Days Forecast', linestyle='dashed', linewidth=1.5, color='red')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Actual vs Predicted Stock Price + Next 7 Days Forecast")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    prediction_chart_path = "static/combined_prediction_forecast.png"
    plt.savefig(prediction_chart_path)
    plt.close()


    # Forecast table
    forecast_table = [(next_7_dates[i].strftime('%Y-%m-%d'), float(next_7_days[i])) for i in range(7)]

    # Evaluation metrics
    mae = mean_absolute_error(y_test[:, 0], y_predicted[:, 0])
    rmse = np.sqrt(mean_squared_error(y_test[:, 0], y_predicted[:, 0]))
    r2 = r2_score(y_test[:, 0], y_predicted[:, 0])

    # Save CSV
    csv_file_path = f"static/{stock}_dataset.csv"
    df.to_csv(csv_file_path)

    return render_template('results.html',
                           plot_path_prediction=prediction_chart_path,
                           plot_path_ema=ema_chart_path,
                           forecast_table=forecast_table,
                           mae=round(mae, 4),
                           rmse=round(rmse, 4),
                           r2=round(r2, 4),
                           data_desc=data_desc.to_html(classes='table table-bordered'),
                           dataset_link=csv_file_path)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
