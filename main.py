#!/usr/bin/env python3
import sys
import os
import argparse
import logging
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Crear directorios necesarios
required_dirs = ['data/historical', 'data/models', 'logs']
for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

try:
    from modules.predictor import ForexPredictor
    from modules.storage import DataStorage
    from modules.dashboard import Dashboard
    from config import CURRENCY_PAIRS, PREDICTION_INTERVALS
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Instalando dependencias...")
    os.system("pip install -r requirements.txt")
    from modules.predictor import ForexPredictor
    from modules.storage import DataStorage
    from modules.dashboard import Dashboard
    from config import CURRENCY_PAIRS, PREDICTION_INTERVALS

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forex_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ForexBot:
    def __init__(self):
        self.predictor = ForexPredictor()
        self.storage = DataStorage()
        
    def run_predictions(self, pairs=None, interval='5m'):
        if pairs is None:
            pairs = CURRENCY_PAIRS
        
        logger.info(f"Iniciando predicciones para {len(pairs)} pares")
        
        self.predictor.initialize_models(pairs)
        predictions = self.predictor.predict_multiple_pairs(pairs, interval)
        
        print("\n" + "="*50)
        print("PREDICCIONES DE FOREX")
        print("="*50)
        
        for prediction in predictions:
            self.print_prediction(prediction)
        
        self.predictor.update_models_with_results()
        logger.info("Proceso completado")
        
        return predictions
    
    def print_prediction(self, prediction):
        symbol = prediction['symbol']
        direction = prediction['direction']
        confidence = prediction['confidence']
        current_price = prediction['current_price']
        price_target = prediction['price_target']
        duration = prediction['duration_minutes']
        
        emoji = "🟢⬆️" if direction == 'UP' else "🔴⬇️" if direction == 'DOWN' else "🟡➡️"
        
        print(f"\n{emoji} {symbol}")
        print(f"   Dirección: {direction}")
        print(f"   Confianza: {confidence:.1%}")
        print(f"   Precio Actual: {current_price:.5f}")
        print(f"   Precio Objetivo: {price_target:.5f}")
        print(f"   Duración: {duration} minutos")
        
        if prediction.get('error'):
            print("   ⚠️  Error en la predicción")
    
    def train_models(self, pairs=None):
        if pairs is None:
            pairs = CURRENCY_PAIRS
        
        results = {}
        for pair in pairs:
            success = self.predictor.train_model(pair)
            results[pair] = success
        
        return results
    
    def show_performance(self):
        summary = self.storage.get_performance_summary()
        
        print("\n" + "="*50)
        print("RENDIMIENTO DE MODELOS")
        print("="*50)
        print(f"Total Predicciones: {summary['total_predictions']}")
        print(f"Precisión General: {summary['overall_accuracy']:.1%}")
        
        if 'by_symbol' in summary:
            for symbol, stats in summary['by_symbol'].items():
                print(f"{symbol}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    def cleanup_old_data(self):
        self.storage.cleanup_old_files()
        print("Datos antiguos limpiados")

def main():
    # Detectar modo dashboard para Streamlit
    if len(sys.argv) > 1 and '--mode' in sys.argv and 'dashboard' in sys.argv:
        print("🖥️  Iniciando Dashboard Web")
        from modules.dashboard import run_dashboard
        run_dashboard()
        return
    
    parser = argparse.ArgumentParser(description='Bot de Predicción de Divisas')
    parser.add_argument('--mode', choices=['predict', 'train', 'dashboard', 'performance', 'cleanup'], 
                       default='dashboard', help='Modo de operación')
    parser.add_argument('--pairs', nargs='+', default=None, help='Pares específicos')
    parser.add_argument('--interval', choices=list(PREDICTION_INTERVALS.keys()), 
                       default='5m', help='Intervalo de predicción')
    
    # Si no hay argumentos, iniciar dashboard
    if len(sys.argv) == 1:
        print("🖥️  Iniciando Dashboard Web")
        from modules.dashboard import run_dashboard
        run_dashboard()
        return
    
    args = parser.parse_args()
    bot = ForexBot()
    
    try:
        if args.mode == 'predict':
            print("🤖 Iniciando Bot de Predicción")
            bot.run_predictions(args.pairs, args.interval)
            
        elif args.mode == 'train':
            print("🎯 Entrenando Modelos")
            bot.train_models(args.pairs)
            
        elif args.mode == 'dashboard':
            print("🖥️  Iniciando Dashboard")
            from modules.dashboard import run_dashboard
            run_dashboard()
            
        elif args.mode == 'performance':
            bot.show_performance()
            
        elif args.mode == 'cleanup':
            bot.cleanup_old_data()
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrumpido por usuario")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
    
    print("\n✅ Finalizado")

if __name__ == "__main__":
    main()
