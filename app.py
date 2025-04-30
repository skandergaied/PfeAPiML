from flask import Flask, jsonify
from flask_cors import CORS  #
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load all datasets
def load_data():
    chicken = pd.read_csv('data/Germany_Chicken_100KG.csv', parse_dates=['Date'])
    wheat = pd.read_csv('data/Germany_Wheat_1Ton.csv', parse_dates=['Date'])
    milk = pd.read_csv('data/Germany_Milk_100KG.csv', parse_dates=['Date'])
    eggs = pd.read_csv('data/Germany_Egg_100KG.csv', parse_dates=['Date'])


    chicken.columns = ['Date', 'Chicken']
    wheat.columns = ['Date', 'Wheat']
    milk.columns = ['Date', 'Milk']
    eggs.columns = ['Date', 'Eggs']

    #Merge datasets
    df = chicken.merge(wheat, on='Date', how='outer')
    df = df.merge(milk, on='Date', how='outer')
    df = df.merge(eggs, on='Date', how='outer')
    df = df.sort_values('Date').set_index('Date')

    # Forward fill missing values
    df = df.ffill()

    return df


# SARIMA
def sarima_forecast(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_steps=12):
    # Fit model
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Forecast
    forecast = model_fit.forecast(steps=forecast_steps)

    return model_fit, forecast


# Evaluate model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f'{model_name} Performance on Test Set:')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}\n')

    return mae, rmse
@app.route("/")
def index():
    return "Hello, your Flask app is live on Render!"

@app.route('/api/predictions')
def get_predictions():
    try:
        df = load_data()
        commodities = ['Chicken', 'Wheat', 'Milk', 'Eggs']
        predictions = {}
        for commodity in commodities:
            series = df[commodity]
            model, forecast = sarima_forecast(series)
            latest_price = series.iloc[-1]
            predicted_price = forecast[-1]
            predictions[commodity] = {
                'latest_price': round(latest_price, 2),
                'predicted_price': round(predicted_price, 2)
            }
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
