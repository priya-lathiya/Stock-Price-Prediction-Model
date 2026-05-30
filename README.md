# Stock Price Prediction Using LSTM

## Overview

This project implements a Long Short-Term Memory (LSTM) Deep Learning model to predict Google (GOOG) stock prices using historical stock market data obtained from Yahoo Finance. The project covers data collection, preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and future stock price forecasting.

A Flask-based web application is integrated to provide an interactive interface for stock price prediction and visualization.

---

## Features

- Historical stock data collection using Yahoo Finance API
- Candlestick chart visualization using Plotly
- Moving Average (MA) and Exponential Moving Average (EMA) analysis
- Data preprocessing and normalization
- Deep Learning-based stock price prediction using LSTM
- Next 7-day stock price forecasting
- Model evaluation using MAE, RMSE, and R² Score
- Interactive Flask web application

---

## Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Plotly
- yFinance
- Scikit-learn
- TensorFlow / Keras
- Flask

### Tools
- Jupyter Notebook
- Visual Studio Code
- Git & GitHub

---

## Project Workflow

### 1. Data Collection

Historical stock price data for Google (GOOG) is fetched using the Yahoo Finance API.

### 2. Data Preprocessing

- Handling and cleaning data
- Date formatting and indexing
- Data normalization using MinMaxScaler
- Train-test split

### 3. Exploratory Data Analysis

Visualizations include:

- Closing Price Trend
- Opening Price Trend
- High Price Trend
- Volume Analysis
- Candlestick Charts

### 4. Technical Indicators

The project computes:

- 100-Day Moving Average (MA100)
- 200-Day Moving Average (MA200)
- 100-Day Exponential Moving Average (EMA100)
- 200-Day Exponential Moving Average (EMA200)

### 5. Model Building

The LSTM network consists of:

- LSTM Layer (50 Units)
- Dropout Layer (20%)
- LSTM Layer (60 Units)
- Dropout Layer (30%)
- LSTM Layer (80 Units)
- Dropout Layer (40%)
- LSTM Layer (120 Units)
- Dropout Layer (50%)
- Dense Output Layer

Optimizer:
- Adam

Loss Function:
- Mean Squared Error (MSE)

### 6. Training & Testing

The dataset is divided into:

- 70% Training Data
- 30% Testing Data

The model learns patterns from historical stock prices and predicts future values.

### 7. Forecasting

The trained model generates:

- Predicted stock prices on test data
- Future stock prices for the next 7 days

### 8. Model Evaluation

Performance metrics used:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Project Structure

```text
Stock-Price-Prediction-Model/
│
├── app.py
├── Google.csv
├── stock_dl_model.h5
├── Stock_Price_Prediction_LSTM.ipynb
├── static/
│   ├── style.css
│   └── assets/
├── templates/
│   └── index.html
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/priya-lathiya/Stock-Price-Prediction-Model.git
cd Stock-Price-Prediction-Model
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib plotly yfinance scikit-learn tensorflow keras flask
```

### Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Results

The LSTM model successfully captures historical stock price patterns and generates future forecasts. The project demonstrates the application of deep learning techniques for financial time-series forecasting.

---

## Future Improvements

- Multi-stock prediction support
- Real-time stock data integration
- Hyperparameter optimization
- Cloud deployment
- Enhanced dashboard and visualizations
- Advanced forecasting models

---

## Notebook

If GitHub notebook preview does not load correctly, view the notebook by downloading the file.
---

## Author

**Priya Lathiya**

B.Tech in Information and Communication Technology

Interests:
- Artificial Intelligence
- Machine Learning
- Data Science
- Deep Learning
- Financial Analytics

GitHub:
https://github.com/priya-lathiya
