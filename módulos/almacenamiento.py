import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import logging
from typing import List, Dict
from config import HISTORICAL_FOLDER, LOGS_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataStorage:
    def __init__(self):
        self.ensure_directories()
    
    def ensure_directories(self):
        """
        Asegura que existan los directorios necesarios
        """
        os.makedirs(HISTORICAL_FOLDER, exist_ok=True)
        os.makedirs(LOGS_FOLDER, exist_ok=True)
    
    def save_historical_data(self, df: pd.DataFrame, symbol: str):
        """
        Guarda datos históricos
        """
        try:
            symbol_clean = symbol.replace('/', '_')
            filepath = os.path.join(HISTORICAL_FOLDER, f'{symbol_clean}_data.parquet')
            
            # Si existe archivo anterior, combinarlo
            if os.path.exists(filepath):
                existing_df = pd.read_parquet(filepath)
                
                # Combinar y eliminar duplicados
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df.drop_duplicates()
                combined_df = combined_df.sort_index()
                
                # Mantener solo últimos 10000 registros
                if len(combined_df) > 10000:
                    combined_df = combined_df.tail(10000)
                
                df_to_save = combined_df
            else:
                df_to_save = df
            
            # Guardar
            df_to_save.to_parquet(filepath, compression='snappy')
            logger.info(f"Datos históricos guardados para {symbol}: {len(df_to_save)} registros")
            
        except Exception as e:
            logger.error(f"Error guardando datos históricos para {symbol}: {e}")
    
    def load_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Carga datos históricos
        """
        try:
            symbol_clean = symbol.replace('/', '_')
            filepath = os.path.join(HISTORICAL_FOLDER, f'{symbol_clean}_data.parquet')
            
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                logger.info(f"Datos históricos cargados para {symbol}: {len(df)} registros")
                return df
            else:
                logger.warning(f"No se encontraron datos históricos para {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error cargando datos históricos para {symbol}: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, prediction: Dict):
        """
        Guarda predicción realizada
        """
        try:
            # Archivo de predicciones por fecha
            today = datetime.now().strftime('%Y-%m-%d')
            filepath = os.path.join(LOGS_FOLDER, f'predictions_{today}.json')
            
            # Preparar datos para JSON
            prediction_json = prediction.copy()
            prediction_json['timestamp'] = prediction_json['timestamp'].isoformat()
            
            # Cargar predicciones existentes
            predictions = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    predictions = json.load(f)
            
            # Agregar nueva predicción
            predictions.append(prediction_json)
            
            # Guardar
            with open(filepath, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            
            logger.info(f"Predicción guardada para {prediction['symbol']}")
            
        except Exception as e:
            logger.error(f"Error guardando predicción: {e}")
    
    def get_past_predictions(self, days_back: int = 7) -> List[Dict]:
        """
        Obtiene predicciones pasadas para evaluación
        """
        try:
            all_predictions = []
            
            for i in range(days_back):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                filepath = os.path.join(LOGS_FOLDER, f'predictions_{date}.json')
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        daily_predictions = json.load(f)
                        
                    # Convertir timestamps
                    for pred in daily_predictions:
                        pred['timestamp'] = datetime.fromisoformat(pred['timestamp'])
                    
                    all_predictions.extend(daily_predictions)
            
            return all_predictions
            
        except Exception as e:
            logger.error(f"Error obteniendo predicciones pasadas: {e}")
            return []
    
    def save_prediction_result(self, prediction: Dict, result: Dict):
        """
        Guarda resultado de evaluación de predicción
        """
        try:
            # Archivo de resultados por mes
            month = datetime.now().strftime('%Y-%m')
            filepath = os.path.join(LOGS_FOLDER, f'results_{month}.json')
            
            # Preparar datos
            result_json = result.copy()
            result_json['prediction_time'] = result_json['prediction_time'].isoformat()
            result_json['evaluation_time'] = datetime.now().isoformat()
            
            # Cargar resultados existentes
            results = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    results = json.load(f)
            
            # Agregar nuevo resultado
            results.append(result_json)
            
            # Guardar
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Resultado guardado para {result['symbol']}")
            
        except Exception as e:
            logger.error(f"Error guardando resultado: {e}")
    
    def get_prediction_results(self, symbol: str, months_back: int = 3) -> List[Dict]:
        """
        Obtiene resultados de predicciones para un símbolo
        """
        try:
            all_results = []
            
            for i in range(months_back):
                date = datetime.now() - timedelta(days=30*i)
                month = date.strftime('%Y-%m')
                filepath = os.path.join(LOGS_FOLDER, f'results_{month}.json')
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        monthly_results = json.load(f)
                    
                    # Filtrar por símbolo
                    symbol_results = [r for r in monthly_results if r['symbol'] == symbol]
                    all_results.extend(symbol_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error obteniendo resultados para {symbol}: {e}")
            return []
    
    def get_performance_summary(self) -> Dict:
        """
        Obtiene resumen de rendimiento general
        """
        try:
            summary = {}
            
            # Obtener todos los resultados recientes
            all_results = []
            for i in range(3):  # Últimos 3 meses
                date = datetime.now() - timedelta(days=30*i)
                month = date.strftime('%Y-%m')
                filepath = os.path.join(LOGS_FOLDER, f'results_{month}.json')
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        monthly_results = json.load(f)
                    all_results.extend(monthly_results)
            
            if not all_results:
                return {'total_predictions': 0, 'overall_accuracy': 0}
            
            # Calcular métricas generales
            total_predictions = len(all_results)
            correct_predictions = sum(1 for r in all_results if r['direction_correct'])
            overall_accuracy = correct_predictions / total_predictions
            
            # Métricas por símbolo
            by_symbol = {}
            for result in all_results:
                symbol = result['symbol']
                if symbol not in by_symbol:
                    by_symbol[symbol] = {'correct': 0, 'total': 0, 'errors': []}
                
                by_symbol[symbol]['total'] += 1
                if result['direction_correct']:
                    by_symbol[symbol]['correct'] += 1
                by_symbol[symbol]['errors'].append(result['price_error'])
            
            # Calcular accuracy por símbolo
            for symbol, stats in by_symbol.items():
                stats['accuracy'] = stats['correct'] / stats['total']
                stats['avg_price_error'] = sum(stats['errors']) / len(stats['errors'])
                del stats['errors']  # No necesario en el resumen
            
            summary = {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'overall_accuracy': overall_accuracy,
                'by_symbol': by_symbol,
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen de rendimiento: {e}")
            return {'total_predictions': 0, 'overall_accuracy': 0}
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """
        Limpia archivos antiguos para mantener el almacenamiento ligero
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Limpiar logs de predicciones
            for filename in os.listdir(LOGS_FOLDER):
                if filename.startswith('predictions_'):
                    try:
                        date_str = filename.replace('predictions_', '').replace('.json', '')
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        
                        if file_date < cutoff_date:
                            filepath = os.path.join(LOGS_FOLDER, filename)
                            os.remove(filepath)
                            logger.info(f"Archivo eliminado: {filename}")
                    except:
                        continue
            
            logger.info(f"Limpieza completada: archivos anteriores a {cutoff_date.date()}")
            
        except Exception as e:
            logger.error(f"Error en limpieza de archivos: {e}")
    
    def export_data_for_analysis(self, symbol: str = None) -> Dict:
        """
        Exporta datos para análisis externo
        """
        try:
            export_data = {}
            
            if symbol:
                symbols_to_export = [symbol]
            else:
                # Obtener todos los símbolos disponibles
                symbols_to_export = []
                for filename in os.listdir(HISTORICAL_FOLDER):
                    if filename.endswith('_data.parquet'):
                        symbol_name = filename.replace('_data.parquet', '').replace('_', '/')
                        symbols_to_export.append(symbol_name)
            
            for sym in symbols_to_export:
                # Datos históricos
                historical_data = self.load_historical_data(sym)
                
                # Resultados de predicciones
                prediction_results = self.get_prediction_results(sym)
                
                export_data[sym] = {
                    'historical_data': historical_data.to_dict('records') if not historical_data.empty else [],
                    'prediction_results': prediction_results,
                    'data_points': len(historical_data),
                    'predictions_evaluated': len(prediction_results)
                }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exportando datos: {e}")
            return {}
