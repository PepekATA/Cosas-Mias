import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
from .data_collector import DataCollector
from .indicators import TechnicalIndicators
from .ml_models import MLModels
from .storage import DataStorage
from config import CURRENCY_PAIRS, PREDICTION_INTERVALS, CONFIDENCE_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexPredictor:
    def __init__(self):
        self.data_collector = DataCollector()
        self.storage = DataStorage()
        self.models = {}  # Un modelo por par de divisas
        
    def initialize_models(self, pairs: List[str] = None):
        """
        Inicializa modelos para los pares especificados
        """
        if pairs is None:
            pairs = CURRENCY_PAIRS
        
        for pair in pairs:
            logger.info(f"Inicializando modelo para {pair}")
            self.models[pair] = MLModels()
            
            # Intentar cargar modelo existente
            if not self.models[pair].load_models(pair):
                # Si no existe, entrenar nuevo modelo
                logger.info(f"Entrenando nuevo modelo para {pair}")
                self.train_model(pair)
    
    def train_model(self, symbol: str):
        """
        Entrena modelo para un par específico
        """
        try:
            # Obtener datos históricos
            df = self.data_collector.get_forex_data(symbol, timeframe='1Min', limit=2000)
            
            if df.empty or not self.data_collector.validate_data(df):
                logger.error(f"Datos inválidos para {symbol}")
                return False
            
            # Calcular indicadores técnicos
            indicators = TechnicalIndicators(df)
            df_with_indicators = indicators.calculate_all_indicators()
            
            # Entrenar modelos
            if symbol not in self.models:
                self.models[symbol] = MLModels()
            
            results = self.models[symbol].train_models(df_with_indicators, symbol)
            
            # Guardar datos históricos
            self.storage.save_historical_data(df_with_indicators, symbol)
            
            logger.info(f"Modelo entrenado exitosamente para {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando modelo para {symbol}: {e}")
            return False
    
    def make_prediction(self, symbol: str, interval: str = '5m') -> Dict:
        """
        Realiza predicción para un par de divisas
        """
        try:
            # Obtener datos recientes
            df = self.data_collector.get_forex_data(symbol, timeframe='1Min', limit=1000)
            
            if df.empty:
                return self._empty_prediction(symbol)
            
            # Calcular indicadores
            indicators = TechnicalIndicators(df)
            df_with_indicators = indicators.calculate_all_indicators()
            
            # Obtener precio actual
            current_price = self.data_collector.get_realtime_price(symbol)
            if current_price is None:
                current_price = df['close'].iloc[-1]
            
            # Realizar predicción con modelos disponibles
            if symbol not in self.models:
                self.models[symbol] = MLModels()
                self.models[symbol].load_models(symbol)
            
            predictions, confidences = self.models[symbol].predict(df_with_indicators)
            
            # Analizar indicadores técnicos para contexto adicional
            technical_analysis = indicators.get_feature_importance_score(df_with_indicators)
            
            # Combinar predicciones
            final_prediction = self._combine_predictions(predictions, confidences, technical_analysis)
            
            # Estimar duración y precio objetivo
            duration_estimate = self._estimate_duration(df_with_indicators, interval)
            price_target = self._estimate_price_target(df_with_indicators, final_prediction['direction'], current_price)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'direction': final_prediction['direction'],  # 'UP' or 'DOWN'
                'confidence': final_prediction['confidence'],
                'duration_minutes': duration_estimate,
                'price_target': price_target,
                'interval': interval,
                'technical_signals': technical_analysis,
                'model_predictions': predictions,
                'model_confidences': confidences
            }
            
            # Guardar predicción
            self.storage.save_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error realizando predicción para {symbol}: {e}")
            return self._empty_prediction(symbol)
    
    def _combine_predictions(self, predictions: Dict, confidences: Dict, technical_analysis: Dict) -> Dict:
        """
        Combina predicciones de múltiples modelos
        """
        if not predictions:
            return {'direction': 'NEUTRAL', 'confidence': 0.5}
        
        # Votos ponderados por confianza
        weighted_votes = 0
        total_weight = 0
        
        for model, prediction in predictions.items():
            confidence = confidences.get(model, 0.5)
            weight = confidence if confidence > CONFIDENCE_THRESHOLD else 0.3
            
            vote = 1 if prediction == 1 else -1
            weighted_votes += vote * weight
            total_weight += weight
        
        # Agregar peso de análisis técnico
        technical_score = sum(technical_analysis.values())
        technical_weight = 0.2
        weighted_votes += technical_score * technical_weight
        total_weight += technical_weight
        
        # Determinar dirección final
        if total_weight > 0:
            final_score = weighted_votes / total_weight
        else:
            final_score = 0
        
        if final_score > 0.1:
            direction = 'UP'
            confidence = min(0.95, 0.5 + abs(final_score))
        elif final_score < -0.1:
            direction = 'DOWN'
            confidence = min(0.95, 0.5 + abs(final_score))
        else:
            direction = 'NEUTRAL'
            confidence = 0.5
        
        return {'direction': direction, 'confidence': confidence}
    
    def _estimate_duration(self, df: pd.DataFrame, interval: str) -> int:
        """
        Estima duración de la tendencia en minutos
        """
        try:
            # Calcular volatilidad promedio
            volatility = df['volatility_5'].iloc[-20:].mean()
            
            # Mapeo base según intervalo
            base_duration = {
                '30s': 2,
                '1m': 5,
                '5m': 15,
                '15m': 45,
                '30m': 90,
                '1h': 180,
                '4h': 720,
                '1d': 1440
            }
            
            base_minutes = base_duration.get(interval, 15)
            
            # Ajustar por volatilidad
            if volatility > 0.001:  # Alta volatilidad
                multiplier = 0.7
            elif volatility < 0.0005:  # Baja volatilidad
                multiplier = 1.3
            else:
                multiplier = 1.0
            
            return int(base_minutes * multiplier)
            
        except:
            return 15  # Valor por defecto
    
    def _estimate_price_target(self, df: pd.DataFrame, direction: str, current_price: float) -> float:
        """
        Estima precio objetivo
        """
        try:
            if direction == 'NEUTRAL':
                return current_price
            
            # Calcular ATR promedio para el rango esperado
            atr = df['ATR'].iloc[-20:].mean()
            
            if pd.isna(atr) or atr == 0:
                # Usar volatilidad alternativa
                volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
                atr = current_price * volatility * 2
            
            # Multiplicador conservador
            multiplier = 0.5
            
            if direction == 'UP':
                target = current_price + (atr * multiplier)
            else:  # DOWN
                target = current_price - (atr * multiplier)
            
            return round(target, 5)
            
        except:
            return current_price
    
    def _empty_prediction(self, symbol: str) -> Dict:
        """
        Retorna predicción vacía en caso de error
        """
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': 0.0,
            'direction': 'NEUTRAL',
            'confidence': 0.5,
            'duration_minutes': 0,
            'price_target': 0.0,
            'interval': 'unknown',
            'technical_signals': {},
            'model_predictions': {},
            'model_confidences': {},
            'error': True
        }
    
    def predict_multiple_pairs(self, pairs: List[str] = None, interval: str = '5m') -> List[Dict]:
        """
        Realiza predicciones para múltiples pares
        """
        if pairs is None:
            pairs = CURRENCY_PAIRS
        
        results = []
        
        for pair in pairs:
            logger.info(f"Realizando predicción para {pair}")
            prediction = self.make_prediction(pair, interval)
            results.append(prediction)
        
        return results
    
    def update_models_with_results(self):
        """
        Actualiza modelos con resultados de predicciones pasadas
        """
        try:
            # Obtener predicciones pasadas para evaluar
            past_predictions = self.storage.get_past_predictions()
            
            for prediction in past_predictions:
                symbol = prediction['symbol']
                predicted_time = prediction['timestamp']
                
                # Verificar si han pasado suficientes minutos para evaluar
                duration = prediction['duration_minutes']
                evaluation_time = predicted_time + timedelta(minutes=duration)
                
                if datetime.now() > evaluation_time:
                    # Obtener precio real en el tiempo de evaluación
                    actual_result = self._evaluate_prediction_accuracy(prediction)
                    
                    if actual_result is not None:
                        # Guardar resultado para aprendizaje futuro
                        self.storage.save_prediction_result(prediction, actual_result)
            
            logger.info("Modelos actualizados con resultados históricos")
            
        except Exception as e:
            logger.error(f"Error actualizando modelos: {e}")
    
    def _evaluate_prediction_accuracy(self, prediction: Dict) -> Dict:
        """
        Evalúa la precisión de una predicción pasada
        """
        try:
            symbol = prediction['symbol']
            predicted_price = prediction['price_target']
            predicted_direction = prediction['direction']
            
            # Obtener precio actual (o más reciente)
            df = self.data_collector.get_forex_data(symbol, timeframe='1Min', limit=10)
            
            if df.empty:
                return None
            
            actual_price = df['close'].iloc[-1]
            initial_price = prediction['current_price']
            
            # Determinar dirección real
            if actual_price > initial_price:
                actual_direction = 'UP'
            elif actual_price < initial_price:
                actual_direction = 'DOWN'
            else:
                actual_direction = 'NEUTRAL'
            
            # Calcular métricas
            direction_correct = predicted_direction == actual_direction
            price_error = abs(actual_price - predicted_price) / initial_price
            
            return {
                'symbol': symbol,
                'prediction_time': prediction['timestamp'],
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'initial_price': initial_price,
                'direction_correct': direction_correct,
                'price_error': price_error,
                'confidence': prediction['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error evaluando predicción: {e}")
            return None
    
    def get_model_performance(self, symbol: str) -> Dict:
        """
        Obtiene estadísticas de rendimiento del modelo
        """
        try:
            results = self.storage.get_prediction_results(symbol)
            
            if not results:
                return {'accuracy': 0, 'total_predictions': 0}
            
            correct_predictions = sum(1 for r in results if r['direction_correct'])
            total_predictions = len(results)
            accuracy = correct_predictions / total_predictions
            
            avg_price_error = sum(r['price_error'] for r in results) / total_predictions
            avg_confidence = sum(r['confidence'] for r in results) / total_predictions
            
            return {
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'avg_price_error': avg_price_error,
                'avg_confidence': avg_confidence,
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo rendimiento para {symbol}: {e}")
            return {'accuracy': 0, 'total_predictions': 0}
