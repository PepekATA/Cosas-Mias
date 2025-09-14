import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
import joblib
import logging
from config import LSTM_LOOKBACK, TRAIN_TEST_SPLIT, MODELS_FOLDER
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModels:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.gru_model = None
        self.rf_model = None
        self.xgb_model = None
        self.lookback = LSTM_LOOKBACK
        
    def prepare_data(self, df):
        """
        Prepara los datos para entrenamiento
        """
        # Seleccionar features importantes
        feature_columns = [
            'open', 'high', 'low', 'close',
            'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'BB_position',
            'EMA_5', 'EMA_10', 'EMA_20',
            'ATR_percent', 'volatility_5',
            'momentum_5', 'momentum_10',
            'trend_short', 'MA_convergence'
        ]
        
        # Filtrar columnas existentes
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if len(available_columns) < 5:
            raise ValueError("Insuficientes features disponibles")
        
        # Preparar dataset
        data = df[available_columns].dropna().copy()
        
        # Crear target (1 si sube, 0 si baja)
        data['future_price'] = data['close'].shift(-1)
        data['target'] = (data['future_price'] > data['close']).astype(int)
        
        # Eliminar última fila (sin target)
        data = data.dropna()
        
        X = data[available_columns].values
        y = data['target'].values
        
        return X, y, available_columns
    
    def create_lstm_sequences(self, X, y):
        """
        Crea secuencias para LSTM
        """
        X_seq, y_seq = [], []
        
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape):
        """
        Construye modelo LSTM
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru_model(self, input_shape):
        """
        Construye modelo GRU
        """
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, df, symbol):
        """
        Entrena todos los modelos
        """
        try:
            X, y, feature_columns = self.prepare_data(df)
            
            if len(X) < 100:
                raise ValueError("Insuficientes datos para entrenamiento")
            
            # Normalizar datos
            X_scaled = self.scaler.fit_transform(X)
            
            # Split train/test
            split_idx = int(len(X_scaled) * TRAIN_TEST_SPLIT)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            results = {}
            
            # 1. Random Forest
            logger.info("Entrenando Random Forest...")
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            self.rf_model.fit(X_train, y_train)
            
            rf_pred = self.rf_model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            results['rf_accuracy'] = rf_accuracy
            
            # 2. XGBoost
            logger.info("Entrenando XGBoost...")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6
            )
            self.xgb_model.fit(X_train, y_train)
            
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            results['xgb_accuracy'] = xgb_accuracy
            
            # 3. LSTM
            if len(X_scaled) > self.lookback * 2:
                logger.info("Entrenando LSTM...")
                X_seq, y_seq = self.create_lstm_sequences(X_scaled, y)
                
                if len(X_seq) > 50:
                    seq_split = int(len(X_seq) * TRAIN_TEST_SPLIT)
                    X_train_seq = X_seq[:seq_split]
                    X_test_seq = X_seq[seq_split:]
                    y_train_seq = y_seq[:seq_split]
                    y_test_seq = y_seq[seq_split:]
                    
                    self.lstm_model = self.build_lstm_model(
                        (self.lookback, X_train_seq.shape[2])
                    )
                    
                    history = self.lstm_model.fit(
                        X_train_seq, y_train_seq,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0
                    )
                    
                    lstm_pred = (self.lstm_model.predict(X_test_seq) > 0.5).astype(int)
                    lstm_accuracy = accuracy_score(y_test_seq, lstm_pred)
                    results['lstm_accuracy'] = lstm_accuracy
            
            # 4. GRU
            if len(X_scaled) > self.lookback * 2:
                logger.info("Entrenando GRU...")
                X_seq, y_seq = self.create_lstm_sequences(X_scaled, y)
                
                if len(X_seq) > 50:
                    self.gru_model = self.build_gru_model(
                        (self.lookback, X_seq.shape[2])
                    )
                    
                    self.gru_model.fit(
                        X_train_seq, y_train_seq,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0
                    )
                    
                    gru_pred = (self.gru_model.predict(X_test_seq) > 0.5).astype(int)
                    gru_accuracy = accuracy_score(y_test_seq, gru_pred)
                    results['gru_accuracy'] = gru_accuracy
            
            # Guardar modelos
            self.save_models(symbol)
            
            logger.info(f"Resultados del entrenamiento para {symbol}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error en entrenamiento para {symbol}: {e}")
            return {}
    
    def predict(self, df):
        """
        Realiza predicción con todos los modelos disponibles
        """
        try:
            X, _, feature_columns = self.prepare_data(df)
            X_scaled = self.scaler.transform(X[-1:])  # Último registro
            
            predictions = {}
            confidences = {}
            
            # Random Forest
            if self.rf_model:
                rf_pred = self.rf_model.predict(X_scaled)[0]
                rf_prob = self.rf_model.predict_proba(X_scaled)[0]
                predictions['rf'] = rf_pred
                confidences['rf'] = max(rf_prob)
            
            # XGBoost
            if self.xgb_model:
                xgb_pred = self.xgb_model.predict(X_scaled)[0]
                xgb_prob = self.xgb_model.predict_proba(X_scaled)[0]
                predictions['xgb'] = xgb_pred
                confidences['xgb'] = max(xgb_prob)
            
            # LSTM
            if self.lstm_model and len(X_scaled) >= self.lookback:
                X_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, -1)
                lstm_prob = self.lstm_model.predict(X_seq)[0][0]
                lstm_pred = int(lstm_prob > 0.5)
                predictions['lstm'] = lstm_pred
                confidences['lstm'] = max(lstm_prob, 1 - lstm_prob)
            
            # GRU
            if self.gru_model and len(X_scaled) >= self.lookback:
                X_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, -1)
                gru_prob = self.gru_model.predict(X_seq)[0][0]
                gru_pred = int(gru_prob > 0.5)
                predictions['gru'] = gru_pred
                confidences['gru'] = max(gru_prob, 1 - gru_prob)
            
            return predictions, confidences
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return {}, {}
    
    def save_models(self, symbol):
        """
        Guarda todos los modelos entrenados
        """
        try:
            symbol_clean = symbol.replace('/', '_')
            
            # Guardar scaler
            joblib.dump(self.scaler, 
                       os.path.join(MODELS_FOLDER, f'scaler_{symbol_clean}.pkl'))
            
            # Guardar modelos sklearn
            if self.rf_model:
                joblib.dump(self.rf_model, 
                           os.path.join(MODELS_FOLDER, f'rf_{symbol_clean}.pkl'))
            
            if self.xgb_model:
                joblib.dump(self.xgb_model, 
                           os.path.join(MODELS_FOLDER, f'xgb_{symbol_clean}.pkl'))
            
            # Guardar modelos de deep learning
            if self.lstm_model:
                self.lstm_model.save(
                    os.path.join(MODELS_FOLDER, f'lstm_{symbol_clean}.h5')
                )
            
            if self.gru_model:
                self.gru_model.save(
                    os.path.join(MODELS_FOLDER, f'gru_{symbol_clean}.h5')
                )
            
            logger.info(f"Modelos guardados para {symbol}")
            
        except Exception as e:
            logger.error(f"Error guardando modelos para {symbol}: {e}")
    
    def load_models(self, symbol):
        """
        Carga modelos previamente entrenados
        """
        try:
            symbol_clean = symbol.replace('/', '_')
            
            # Cargar scaler
            scaler_path = os.path.join(MODELS_FOLDER, f'scaler_{symbol_clean}.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Cargar Random Forest
            rf_path = os.path.join(MODELS_FOLDER, f'rf_{symbol_clean}.pkl')
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
            
            # Cargar XGBoost
            xgb_path = os.path.join(MODELS_FOLDER, f'xgb_{symbol_clean}.pkl')
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
            
            # Cargar LSTM
            lstm_path = os.path.join(MODELS_FOLDER, f'lstm_{symbol_clean}.h5')
            if os.path.exists(lstm_path):
                self.lstm_model = tf.keras.models.load_model(lstm_path)
            
            # Cargar GRU
            gru_path = os.path.join(MODELS_FOLDER, f'gru_{symbol_clean}.h5')
            if os.path.exists(gru_path):
                self.gru_model = tf.keras.models.load_model(gru_path)
            
            logger.info(f"Modelos cargados para {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelos para {symbol}: {e}")
            return False
