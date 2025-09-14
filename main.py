#!/usr/bin/env python3
"""
Bot de Predicción de Divisas - Punto de entrada principal
ACTUALIZADO PARA DEPLOYMENT
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Crear directorios necesarios si no existen
required_dirs = ['data/historical', 'data/models', 'logs']
for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

try:
    from modules import ForexPredictor, DataStorage, Dashboard
    from config import CURRENCY_PAIRS, PREDICTION_INTERVALS
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Instalando dependencias faltantes...")
    os.system("pip install -r requirements.txt")
    from modules import ForexPredictor, DataStorage, Dashboard
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
        """
        Ejecuta predicciones para los pares especificados
        """
        if pairs is None:
            pairs = CURRENCY_PAIRS
        
        logger.info(f"Iniciando predicciones para {len(pairs)} pares")
        logger.info(f"Pares: {pairs}")
        logger.info(f"Intervalo: {interval}")
        
        # Inicializar modelos
        self.predictor.initialize_models(pairs)
        
        # Generar predicciones
        predictions = self.predictor.predict_multiple_pairs(pairs, interval)
        
        # Mostrar resultados
        print("\n" + "="*50)
        print("PREDICCIONES DE FOREX")
        print("="*50)
        
        for prediction in predictions:
            self.print_prediction(prediction)
        
        # Actualizar modelos con resultados pasados
        logger.info("Actualizando modelos con resultados históricos...")
        self.predictor.update_models_with_results()
        
        logger.info("Proceso completado exitosamente")
        
        return predictions
    
    def print_prediction(self, prediction):
        """
        Imprime una predicción de forma legible
        """
        symbol = prediction['symbol']
        direction = prediction['direction']
        confidence = prediction['confidence']
        current_price = prediction['current_price']
        price_target = prediction['price_target']
        duration = prediction['duration_minutes']
        
        # Emojis según dirección
        emoji = "🟢⬆️" if direction == 'UP' else "🔴⬇️" if direction == 'DOWN' else "🟡➡️"
        
        print(f"\n{emoji} {symbol}")
        print(f"   Dirección: {direction}")
        print(f"   Confianza: {confidence:.1%}")
        print(f"   Precio Actual: {current_price:.5f}")
        print(f"   Precio Objetivo: {price_target:.5f}")
        print(f"   Duración Estimada: {duration} minutos")
        
        if prediction.get('error'):
            print("   ⚠️  Error en la predicción")
    
    # ... resto del código igual ...

def main():
    # Detectar si se está ejecutando desde Streamlit
    if len(sys.argv) > 1 and sys.argv[1] == '--mode' and sys.argv[2] == 'dashboard':
        print("🖥️  Iniciando Dashboard Web")
        try:
            from modules.dashboard import run_dashboard
            run_dashboard()
        except Exception as e:
            print(f"Error iniciando dashboard: {e}")
            # Fallback básico
            import streamlit as st
            st.title("Error en Dashboard")
            st.error(f"Error: {e}")
        return
    
    # Resto de la lógica original
    parser = argparse.ArgumentParser(description='Bot de Predicción de Divisas')
    parser.add_argument('--mode', choices=['predict', 'train', 'dashboard', 'performance', 'cleanup'], 
                       default='dashboard', help='Modo de operación')
    parser.add_argument('--pairs', nargs='+', default=None, 
                       help='Pares de divisas a procesar')
    parser.add_argument('--interval', choices=list(PREDICTION_INTERVALS.keys()), 
                       default='5m', help='Intervalo de predicción')
    
    # Si no hay argumentos, ejecutar dashboard por defecto
    if len(sys.argv) == 1:
        print("🖥️  Iniciando Dashboard Web (modo por defecto)")
        try:
            from modules.dashboard import run_dashboard
            run_dashboard()
        except Exception as e:
            print(f"Error: {e}")
        return
    
    args = parser.parse_args()
    
    # Crear instancia del bot
    bot = ForexBot()
    
    try:
        if args.mode == 'predict':
            # Modo predicción
            print("🤖 Iniciando Bot de Predicción de Divisas")
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            predictions = bot.run_predictions(args.pairs, args.interval)
            
        elif args.mode == 'train':
            # Modo entrenamiento
            print("🎯 Iniciando Entrenamiento de Modelos")
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            results = bot.train_models(args.pairs)
            
        elif args.mode == 'dashboard':
            # Modo dashboard
            print("🖥️  Iniciando Dashboard Web")
            
            from modules.dashboard import run_dashboard
            run_dashboard()
            
        elif args.mode == 'performance':
            # Mostrar rendimiento
            bot.show_performance()
            
        elif args.mode == 'cleanup':
            # Limpiar datos
            bot.cleanup_old_data()
    
    except KeyboardInterrupt:
        print("\n⏹️  Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        print(f"❌ Error: {e}")
    
    print("\n✅ Proceso finalizado")

if __name__ == "__main__":
    main()
