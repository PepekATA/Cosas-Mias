import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            ALPACA_BASE_URL,
            api_version='v2'
        )
    
    def get_forex_data(self, symbol, timeframe='1Min', limit=1000):
        """
        Obtiene datos de forex desde Alpaca API
        """
        try:
            # Convertir símbolo al formato de Alpaca
            alpaca_symbol = symbol.replace('/', '')
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            bars = self.api.get_bars(
                alpaca_symbol,
                timeframe,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=limit
            )
            
            df = pd.DataFrame([{
                'timestamp': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            } for bar in bars])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Datos obtenidos para {symbol}: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de Alpaca para {symbol}: {e}")
            return self._get_yahoo_data(symbol, limit)
    
    def _get_yahoo_data(self, symbol, limit):
        """
        Alternativa usando Yahoo Finance
        """
        try:
            yahoo_symbol = symbol + '=X'
            ticker = yf.Ticker(yahoo_symbol)
            
            df = ticker.history(period='30d', interval='1m')
            
            if df.empty:
                raise Exception(f"No hay datos disponibles para {symbol}")
            
            df.reset_index(inplace=True)
            df.rename(columns={
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            df.set_index('timestamp', inplace=True)
            
            # Limitar registros
            df = df.tail(limit)
            
            logger.info(f"Datos alternativos obtenidos para {symbol}: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos alternativos para {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_price(self, symbol):
        """
        Obtiene precio en tiempo real
        """
        try:
            alpaca_symbol = symbol.replace('/', '')
            latest_trade = self.api.get_latest_trade(alpaca_symbol)
            return latest_trade.price
        except:
            # Alternativa con Yahoo
            try:
                yahoo_symbol = symbol + '=X'
                ticker = yf.Ticker(yahoo_symbol)
                data = ticker.history(period='1d', interval='1m')
                return data['Close'].iloc[-1]
            except:
                logger.error(f"No se pudo obtener precio en tiempo real para {symbol}")
                return None
    
    def validate_data(self, df):
        """
        Valida la calidad de los datos
        """
        if df.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Verificar datos faltantes
        if df[required_columns].isnull().any().any():
            logger.warning("Datos faltantes detectados")
        
        # Verificar valores negativos
        if (df[required_columns] <= 0).any().any():
            logger.warning("Valores negativos detectados")
        
        return True
