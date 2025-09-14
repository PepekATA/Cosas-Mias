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
        os.makedirs(HISTORICAL_FOLDER, exist_ok=True)
        os.makedirs(LOGS_FOLDER, exist_ok=True)
    
    def save_historical_data(self, df: pd.DataFrame, symbol: str):
        try:
            symbol_clean = symbol.replace('/', '_')
            filepath = os.path.join(HISTORICAL_FOLDER, f'{symbol_clean}_data.parquet')
            
            if os.path.exists(filepath):
                existing_df = pd.read_parquet(filepath)
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df.drop_duplicates()
                combined_df = combined_df.sort_index()
                
                if len(combined_df) > 10000:
                    combined_df = combined_df.tail(10000)
                
                df_to_save = combined_df
            else:
                df_to_save = df
            
            df_to_save.to_parquet(filepath, compression='snappy')
            logger.info(f"Datos guardados para {symbol}: {len(df_to_save)} registros")
            
        except Exception as e:
            logger.error(f"Error guardando datos para {symbol}: {e}")
    
    def load_historical_data(self, symbol: str) -> pd.DataFrame:
        try:
            symbol_clean = symbol.replace('/', '_')
            filepath = os.path.join(HISTORICAL_FOLDER, f'{symbol_clean}_data.parquet')
            
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error cargando datos para {symbol}: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, prediction: Dict):
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            filepath = os.path.join(LOGS_FOLDER, f'predictions_{today}.json')
            
            prediction_json = prediction.copy()
            if isinstance(prediction_json['timestamp'], datetime):
                prediction_json['timestamp'] = prediction_json['timestamp'].isoformat()
            
            predictions = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    predictions = json.load(f)
            
            predictions.append(prediction_json)
            
            with open(filepath, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando predicción: {e}")
    
    def get_past_predictions(self, days_back: int = 7) -> List[Dict]:
        try:
            all_predictions = []
            
            for i in range(days_back):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                filepath = os.path.join(LOGS_FOLDER, f'predictions_{date}.json')
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        daily_predictions = json.load(f)
                        
                    for pred in daily_predictions:
                        pred['timestamp'] = datetime.fromisoformat(pred['timestamp'])
                    
                    all_predictions.extend(daily_predictions)
            
            return all_predictions
            
        except Exception as e:
            logger.error(f"Error obteniendo predicciones pasadas: {e}")
            return []
    
    def save_prediction_result(self, prediction: Dict, result: Dict):
        try:
            month = datetime.now().strftime('%Y-%m')
            filepath = os.path.join(LOGS_FOLDER, f'results_{month}.json')
            
            result_json = result.copy()
            if isinstance(result_json['prediction_time'], datetime):
                result_json['prediction_time'] = result_json['prediction_time'].isoformat()
            result_json['evaluation_time'] = datetime.now().isoformat()
            
            results = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    results = json.load(f)
            
            results.append(result_json)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando resultado: {e}")
    
    def get_prediction_results(self, symbol: str, months_back: int = 3) -> List[Dict]:
        try:
            all_results = []
            
            for i in range(months_back):
                date = datetime.now() - timedelta(days=30*i)
                month = date.strftime('%Y-%m')
                filepath = os.path.join(LOGS_FOLDER, f'results_{month}.json')
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        monthly_results = json.load(f)
                    
                    symbol_results = [r for r in monthly_results if r['symbol'] == symbol]
                    all_results.extend(symbol_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error obteniendo resultados para {symbol}: {e}")
            return []
    
    def get_performance_summary(self) -> Dict:
        try:
            all_results = []
            for i in range(3):
                date = datetime.now() - timedelta(days=30*i)
                month = date.strftime('%Y-%m')
                filepath = os.path.join(LOGS_FOLDER, f'results_{month}.json')
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        monthly_results = json.load(f)
                    all_results.extend(monthly_results)
            
            if not all_results:
                return {'total_predictions': 0, 'overall_accuracy': 0}
            
            total_predictions = len(all_results)
            correct_predictions = sum(1 for r in all_results if r.get('direction_correct', False))
            overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            by_symbol = {}
            for result in all_results:
                symbol = result['symbol']
                if symbol not in by_symbol:
                    by_symbol[symbol] = {'correct': 0, 'total': 0, 'errors': []}
                
                by_symbol[symbol]['total'] += 1
                if result.get('direction_correct', False):
                    by_symbol[symbol]['correct'] += 1
                by_symbol[symbol]['errors'].append(result.get('price_error', 0))
            
            for symbol, stats in by_symbol.items():
                stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                stats['avg_price_error'] = sum(stats['errors']) / len(stats['errors']) if stats['errors'] else 0
                del stats['errors']
            
            return {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'overall_accuracy': overall_accuracy,
                'by_symbol': by_symbol,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen: {e}")
            return {'total_predictions': 0, 'overall_accuracy': 0}
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for filename in os.listdir(LOGS_FOLDER):
                if filename.startswith('predictions_'):
                    try:
                        date_str = filename.replace('predictions_', '').replace('.json', '')
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        
                        if file_date < cutoff_date:
                            filepath = os.path.join(LOGS_FOLDER, filename)
                            os.remove(filepath)
                    except:
                        continue
            
        except Exception as e:
            logger.error(f"Error en limpieza: {e}")
