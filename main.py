#!/usr/bin/env python3
"""
Bot de Predicción de Divisas - Punto de entrada principal
Versión completa y funcional
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Crear directorios necesarios si no existen
required_dirs = ['data/historical', 'data/models', 'logs']
for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

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

# Importar configuración
try:
    from config import CURRENCY_PAIRS, PREDICTION_INTERVALS
except ImportError:
    print("❌ Error: No se encontró config.py")
    sys.exit(1)

# Función para instalar dependencias faltantes
def install_dependencies():
    """Instala dependencias faltantes"""
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencias instaladas correctamente")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        sys.exit(1)

# Importar módulos con manejo de errores
try:
    from modules.predictor import ForexPredictor
    from modules.storage import DataStorage
    from modules.dashboard import Dashboard
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    print("📦 Instalando dependencias faltantes...")
    install_dependencies()
    
    # Intentar importar de nuevo
    try:
        from modules.predictor import ForexPredictor
        from modules.storage import DataStorage
        from modules.dashboard import Dashboard
    except ImportError as e:
        print(f"❌ Error crítico importando módulos: {e}")
        print("🔧 Verifica que todos los archivos estén en su lugar")
        sys.exit(1)

class ForexBot:
    """Clase principal del bot de predicción de divisas"""
    
    def __init__(self):
        """Inicializa el bot"""
        try:
            self.predictor = ForexPredictor()
            self.storage = DataStorage()
            logger.info("Bot inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando bot: {e}")
            raise
    
    def run_predictions(self, pairs=None, interval='5m'):
        """
        Ejecuta predicciones para los pares especificados
        """
        if pairs is None:
            pairs = CURRENCY_PAIRS[:4]  # Solo primeros 4 para prueba
        
        logger.info(f"🚀 Iniciando predicciones para {len(pairs)} pares")
        logger.info(f"📊 Pares: {pairs}")
        logger.info(f"⏰ Intervalo: {interval}")
        
        try:
            # Inicializar modelos
            print("🤖 Inicializando modelos...")
            self.predictor.initialize_models(pairs)
            
            # Generar predicciones
            print("📈 Generando predicciones...")
            predictions = self.predictor.predict_multiple_pairs(pairs, interval)
            
            # Mostrar resultados
            self.display_results(predictions)
            
            # Actualizar modelos con resultados pasados
            print("🔄 Actualizando modelos...")
            self.predictor.update_models_with_results()
            
            logger.info("✅ Proceso de predicción completado exitosamente")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error en predicciones: {e}")
            print(f"❌ Error: {e}")
            return []
    
    def display_results(self, predictions):
        """Muestra los resultados de las predicciones de forma legible"""
        print("\n" + "="*60)
        print("🎯 PREDICCIONES DE FOREX - RESULTADOS")
        print("="*60)
        print(f"⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Total de predicciones: {len(predictions)}")
        print("="*60)
        
        if not predictions:
            print("⚠️  No se generaron predicciones")
            return
        
        for i, prediction in enumerate(predictions, 1):
            self.print_single_prediction(prediction, i)
        
        print("="*60)
        print("✅ Todas las predicciones completadas")
    
    def print_single_prediction(self, prediction, index):
        """Imprime una predicción individual de forma elegante"""
        symbol = prediction.get('symbol', 'N/A')
        direction = prediction.get('direction', 'UNKNOWN')
        confidence = prediction.get('confidence', 0.0)
        current_price = prediction.get('current_price', 0.0)
        price_target = prediction.get('price_target', 0.0)
        duration = prediction.get('duration_minutes', 0)
        
        # Emojis según dirección
        if direction == 'UP':
            emoji = "🟢⬆️"
            color = "VERDE"
        elif direction == 'DOWN':
            emoji = "🔴⬇️"
            color = "ROJO"
        else:
            emoji = "🟡➡️"
            color = "AMARILLO"
        
        # Calcular cambio porcentual esperado
        if current_price > 0:
            pct_change = ((price_target - current_price) / current_price) * 100
        else:
            pct_change = 0
        
        print(f"\n📈 PREDICCIÓN #{index}")
        print(f"   {emoji} Par: {symbol}")
        print(f"   🎯 Dirección: {direction} ({color})")
        print(f"   📊 Confianza: {confidence:.1%}")
        print(f"   💰 Precio Actual: {current_price:.5f}")
        print(f"   🎪 Precio Objetivo: {price_target:.5f}")
        print(f"   📈 Cambio Esperado: {pct_change:+.3f}%")
        print(f"   ⏱️  Duración Estimada: {duration} minutos")
        
        # Mostrar advertencias si hay errores
        if prediction.get('error'):
            print(f"   ⚠️  ADVERTENCIA: Error en la predicción")
        
        # Mostrar nivel de confianza con barras
        confidence_bars = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        print(f"   📊 Confianza: [{confidence_bars}] {confidence:.1%}")
    
    def train_models(self, pairs=None):
        """Entrena modelos para los pares especificados"""
        if pairs is None:
            pairs = CURRENCY_PAIRS[:4]
        
        print("🎯 INICIANDO ENTRENAMIENTO DE MODELOS")
        print("="*50)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Pares a entrenar: {pairs}")
        print("="*50)
        
        results = {}
        successful = 0
        
        for i, pair in enumerate(pairs, 1):
            print(f"\n🔄 Entrenando modelo {i}/{len(pairs)}: {pair}")
            try:
                success = self.predictor.train_model(pair)
                results[pair] = success
                
                if success:
                    successful += 1
                    print(f"   ✅ {pair}: Entrenamiento exitoso")
                else:
                    print(f"   ❌ {pair}: Falló el entrenamiento")
                    
            except Exception as e:
                print(f"   ❌ {pair}: Error - {e}")
                results[pair] = False
        
        # Resumen final
        print("\n" + "="*50)
        print("📊 RESUMEN DE ENTRENAMIENTO")
        print("="*50)
        print(f"✅ Exitosos: {successful}/{len(pairs)} ({successful/len(pairs)*100:.1f}%)")
        print(f"❌ Fallidos: {len(pairs) - successful}/{len(pairs)}")
        
        for pair, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {pair}")
        
        return results
    
    def show_performance(self):
        """Muestra estadísticas de rendimiento de los modelos"""
        print("📊 RENDIMIENTO DE MODELOS")
        print("="*50)
        
        try:
            summary = self.storage.get_performance_summary()
            
            if summary['total_predictions'] == 0:
                print("⚠️  No hay datos de rendimiento disponibles")
                print("💡 Ejecuta algunas predicciones primero")
                return
            
            # Estadísticas generales
            print(f"📈 Total de Predicciones: {summary['total_predictions']}")
            print(f"✅ Predicciones Correctas: {summary['correct_predictions']}")
            print(f"🎯 Precisión General: {summary['overall_accuracy']:.1%}")
            print(f"🕐 Última Actualización: {summary.get('last_updated', 'N/A')}")
            
            # Rendimiento por símbolo
            if 'by_symbol' in summary and summary['by_symbol']:
                print("\n📊 RENDIMIENTO POR PAR:")
                print("-" * 50)
                
                for symbol, stats in summary['by_symbol'].items():
                    accuracy = stats['accuracy']
                    total = stats['total']
                    correct = stats['correct']
                    
                    # Barra de progreso visual
                    progress_bar = "█" * int(accuracy * 20) + "░" * (20 - int(accuracy * 20))
                    
                    print(f"   {symbol}:")
                    print(f"     Precisión: [{progress_bar}] {accuracy:.1%}")
                    print(f"     Resultados: {correct}/{total}")
                    print(f"     Error Promedio: {stats.get('avg_price_error', 0):.4f}")
                    print()
            
        except Exception as e:
            logger.error(f"Error mostrando rendimiento: {e}")
            print(f"❌ Error obteniendo estadísticas: {e}")
    
    def cleanup_old_data(self):
        """Limpia datos antiguos para mantener el sistema ligero"""
        print("🧹 LIMPIEZA DE DATOS ANTIGUOS")
        print("="*40)
        
        try:
            self.storage.cleanup_old_files()
            print("✅ Limpieza completada exitosamente")
            print("💾 Espacio liberado en disco")
            
        except Exception as e:
            logger.error(f"Error en limpieza: {e}")
            print(f"❌ Error durante la limpieza: {e}")

def run_dashboard_mode():
    """Ejecuta el dashboard web de Streamlit"""
    print("🖥️  INICIANDO DASHBOARD WEB")
    print("="*40)
    print("🌐 Abriendo interfaz web...")
    print("📱 Accede desde tu navegador")
    print("⏹️  Presiona Ctrl+C para detener")
    print("="*40)
    
    try:
        from modules.dashboard import run_dashboard
        run_dashboard()
    except ImportError:
        print("❌ Error: Módulo dashboard no encontrado")
        print("🔧 Verifica que modules/dashboard.py existe")
    except Exception as e:
        logger.error(f"Error iniciando dashboard: {e}")
        print(f"❌ Error iniciando dashboard: {e}")
        
        # Fallback con Streamlit básico
        try:
            import streamlit as st
            st.title("❌ Error en Dashboard")
            st.error(f"No se pudo inicializar el dashboard: {e}")
            st.info("Verifica que todos los módulos estén instalados correctamente")
        except ImportError:
            print("❌ Streamlit no está instalado")
            print("📦 Instala con: pip install streamlit")

def display_banner():
    """Muestra banner de bienvenida"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🤖 BOT DE PREDICCIÓN DE DIVISAS 🤖                ║
    ║                                                              ║
    ║                  📈 FOREX PREDICTOR v2.0                    ║
    ║                                                              ║
    ║               Predicciones inteligentes con IA               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Sistema: Listo y operativo")
    print()

def main():
    """Función principal del programa"""
    
    # Mostrar banner
    display_banner()
    
    # Detectar si se ejecuta desde Streamlit con argumentos específicos
    if len(sys.argv) > 1 and any('--mode' in arg for arg in sys.argv):
        if 'dashboard' in sys.argv:
            run_dashboard_mode()
            return
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(
        description='🤖 Bot de Predicción de Divisas con IA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
    python main.py                                    # Dashboard web (por defecto)
    python main.py --mode predict                     # Predicciones básicas
    python main.py --mode predict --pairs EUR/USD    # Predicción específica
    python main.py --mode train                       # Entrenar modelos
    python main.py --mode performance                 # Ver rendimiento
    python main.py --mode cleanup                     # Limpiar datos antiguos
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['predict', 'train', 'dashboard', 'performance', 'cleanup'], 
        default='dashboard',
        help='Modo de operación del bot'
    )
    
    parser.add_argument(
        '--pairs', 
        nargs='+', 
        default=None,
        help='Pares de divisas específicos (ej: EUR/USD GBP/USD)'
    )
    
    parser.add_argument(
        '--interval', 
        choices=list(PREDICTION_INTERVALS.keys()), 
        default='5m',
        help='Intervalo de tiempo para predicción'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Modo verbose para más detalles'
    )
    
    # Si no hay argumentos, ejecutar dashboard por defecto
    if len(sys.argv) == 1:
        run_dashboard_mode()
        return
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Configurar nivel de logging si verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Modo verbose activado")
    
    # Crear instancia del bot
    try:
        bot = ForexBot()
    except Exception as e:
        print(f"❌ Error crítico inicializando bot: {e}")
        logger.error(f"Error crítico: {e}")
        return
    
    # Ejecutar según el modo seleccionado
    try:
        if args.mode == 'predict':
            print("🤖 MODO: PREDICCIÓN")
            predictions = bot.run_predictions(args.pairs, args.interval)
            
            if predictions:
                print(f"\n✅ Se generaron {len(predictions)} predicciones")
            else:
                print("\n⚠️  No se pudieron generar predicciones")
                
        elif args.mode == 'train':
            print("🎯 MODO: ENTRENAMIENTO")
            results = bot.train_models(args.pairs)
            
            success_count = sum(results.values()) if results else 0
            total_count = len(results) if results else 0
            
            if success_count == total_count and total_count > 0:
                print(f"\n✅ Todos los modelos entrenados exitosamente ({success_count}/{total_count})")
            else:
                print(f"\n⚠️  Entrenamiento completado: {success_count}/{total_count} exitosos")
                
        elif args.mode == 'dashboard':
            print("🖥️  MODO: DASHBOARD WEB")
            run_dashboard_mode()
            
        elif args.mode == 'performance':
            print("📊 MODO: ANÁLISIS DE RENDIMIENTO")
            bot.show_performance()
            
        elif args.mode == 'cleanup':
            print("🧹 MODO: LIMPIEZA DE DATOS")
            bot.cleanup_old_data()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  PROCESO INTERRUMPIDO POR EL USUARIO")
        print("👋 ¡Hasta luego!")
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        print(f"\n❌ ERROR DURANTE LA EJECUCIÓN: {e}")
        print("🔧 Revisa los logs para más detalles")
    
    finally:
        print("\n" + "="*60)
        print("✅ PROCESO FINALIZADO")
        print(f"⏰ Terminado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📝 Revisa los logs en: logs/forex_bot.log")
        print("="*60)

if __name__ == "__main__":
    main()
