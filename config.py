import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de APIs
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'your_alpaca_key_here')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'your_alpaca_secret_here')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Para pruebas

# Configuración de datos
DATA_FOLDER = 'data'
HISTORICAL_FOLDER = os.path.join(DATA_FOLDER, 'historical')
MODELS_FOLDER = os.path.join(DATA_FOLDER, 'models')
LOGS_FOLDER = 'logs'

# Pares de divisas principales
CURRENCY_PAIRS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
    'AUD/USD', 'USD/CAD', 'NZD/USD'
]

# Configuración de predicción
PREDICTION_INTERVALS = {
    '30s': 30,
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400
}

# Parámetros de modelos ML
LSTM_LOOKBACK = 60
TRAIN_TEST_SPLIT = 0.8
CONFIDENCE_THRESHOLD = 0.6

# Crear directorios si no existen
os.makedirs(HISTORICAL_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
