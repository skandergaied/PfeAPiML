from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
import logging
import pickle
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warning messages
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create cache directory if it doesn't exist
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Memory-efficient data loading
@lru_cache(maxsize=1)
def load_data():
    """Load and merge all datasets with caching to avoid reloading"""
    try:
        logger.info("Loading datasets...")
        
        chicken = pd.read_csv('data/Germany_Chicken_100KG.csv', parse_dates=['Date'])
        wheat = pd.read_csv('data/Germany_Wheat_1Ton.csv', parse_dates=['Date'])
        milk = pd.read_csv('data/Germany_Milk_100KG.csv', parse_dates=['Date'])
        eggs = pd.read_csv('data/Germany_Egg_100KG.csv', parse_dates=['Date'])
        
        chicken.columns = ['Date', 'Chicken']
        wheat.columns = ['Date', 'Wheat']
        milk.columns = ['Date', 'Milk']
        eggs.columns = ['Date', 'Eggs']
        
        # Merge datasets
        df = chicken.merge(wheat, on='Date', how='outer')
        df = df.merge(milk, on='Date', how='outer')
        df = df.merge(eggs, on='Date', how='outer')
        df = df.sort_values('Date').set_index('Date')
        
        # Forward fill missing values
        df = df.ffill()
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def get_cached_model_path(commodity):
    """Get path for cached model"""
    return os.path.join(CACHE_DIR, f"{commodity}_model.pkl")

def check_model_cache(commodity):
    """Check if model exists in cache and is recent (less than 24 hours old)"""
    cache_path = get_cached_model_path(commodity)
    if os.path.exists(cache_path):
        # Check if cache is less than 24 hours old
        modification_time = os.path.getmtime(cache_path)
        if (time.time() - modification_time) < 86400:  # 24 hours in seconds
            return True
    return False

def save_model_to_cache(commodity, model):
    """Save model to cache"""
    try:
        cache_path = get_cached_model_path(commodity)
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model for {commodity} saved to cache")
    except Exception as e:
        logger.warning(f"Failed to cache model for {commodity}: {str(e)}")

def load_model_from_cache(commodity):
    """Load model from cache"""
    try:
        cache_path = get_cached_model_path(commodity)
        with open(cache_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model for {commodity} loaded from cache")
        return model
    except Exception as e:
        logger.warning(f"Failed to load cached model for {commodity}: {str(e)}")
        return None

def sarima_forecast(series, commodity, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_steps=12, use_cache=True):
    """Fit SARIMA model and generate forecasts with caching support"""
    # Try to use cached model if available and requested
    if use_cache and check_model_cache(commodity):
        try:
            model_fit = load_model_from_cache(commodity)
            # Generate forecast using the cached model
            forecast = model_fit.forecast(steps=forecast_steps)
            return model_fit, forecast
        except Exception as e:
            logger.warning(f"Error using cached model for {commodity}: {str(e)}. Retraining...")
    
    # Train new model if cache isn't available or failed
    try:
        logger.info(f"Training new SARIMA model for {commodity}...")
        
        # Use a more memory-efficient approach for model fitting
        # Start with a simple model and gradually increase complexity if needed
        try:
            # Try a simpler model first
            model = SARIMAX(series, order=(1, 1, 0), seasonal_order=(0, 1, 0, 12),
                          enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False, low_memory=True)
        except Exception as e:
            logger.warning(f"Simple model failed for {commodity}, trying specified model: {str(e)}")
            # Fall back to the specified model parameters
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False, low_memory=True)
        
        # Forecast
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Cache the model
        save_model_to_cache(commodity, model_fit)
        
        return model_fit, forecast
    
    except Exception as e:
        logger.error(f"SARIMA model failed for {commodity}: {str(e)}")
        raise

@app.route('/api/predictions', methods=["GET"])
def get_predictions():
    start_time = time.time()
    try:
        df = load_data()
        commodities = ['Chicken', 'Wheat', 'Milk', 'Eggs']
        predictions = {}
        
        for commodity in commodities:
            try:
                logger.info(f"Processing {commodity}...")
                series = df[commodity]
                
                # Get model and forecast
                model, forecast = sarima_forecast(series, commodity)
                
                latest_price = series.iloc[-1]
                predicted_price = forecast[-1]
                
                predictions[commodity] = {
                    'latest_price': round(float(latest_price), 2),
                    'predicted_price': round(float(predicted_price), 2),
                    'change_percentage': round(100 * (predicted_price - latest_price) / latest_price, 2)
                }
                
            except Exception as e:
                logger.error(f"Error processing {commodity}: {str(e)}")
                predictions[commodity] = {
                    'error': f"Failed to generate prediction: {str(e)}"
                }
        
        execution_time = round(time.time() - start_time, 2)
        logger.info(f"Predictions generated in {execution_time} seconds")
        
        return jsonify({
            'predictions': predictions,
            'execution_time_seconds': execution_time
        })
    
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
