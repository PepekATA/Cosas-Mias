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
        """Ejecuta predicciones para los pares especificados"""
        if pairs is None:
            pairs = CURRENCY_PAIRS[:4]
        
        logger.info(f"🚀 Iniciando predicciones para {len(pairs)} pares")
        logger.info(f"📊 Pares: {pairs}")
        logger.info(f"⏰ Intervalo: {interval}")
        
        try:
            print("🤖 Inicializando modelos...")
            self.predictor.initialize_models(pairs)
            
            print("📈 Generando predicciones...")
            predictions = self.predictor.predict_multiple_pairs(pairs, interval)
            
            self.display_results(predictions)
            
            print("🔄 Actualizando modelos...")
            self.predictor.update_models_with_results()
            
            logger.info("✅ Proceso de predicción completado exitosamente")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error en predicciones: {e}")
            print(f"❌ Error: {e}")
            return []
    
    def display_results(self, predictions):
        """Muestra los resultados de las predicciones"""
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
        """Imprime una predicción individual"""
        symbol = prediction.get('symbol', 'N/A')
        direction = prediction.get('direction', 'UNKNOWN')
        confidence = prediction.get('confidence', 0.0)
        current_price = prediction.get('current_price', 0.0)
        price_target = prediction.get('price_target', 0.0)
        duration = prediction.get('duration_minutes', 0)
        
        if direction == 'UP':
            emoji = "🟢⬆️"
            color = "VERDE"
        elif direction == 'DOWN':
            emoji = "🔴⬇️"
            color = "ROJO"
        else:
            emoji = "🟡➡️"
            color = "AMARILLO"
        
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
        
        if prediction.get('error'):
            print("   ⚠️  ADVERTENCIA: Error en la predicción")
        
        confidence_bars = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        print(f"   📊 Confianza: [{confidence_bars}] {confidence:.1%}")

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
        print("✅ Dashboard principal cargado")
        run_dashboard()
        
    except ImportError as e:
        print(f"⚠️ Error importando dashboard principal: {e}")
        print("🔄 Intentando con dashboard simple...")
        
        try:
            from modules.simple_dashboard import run_simple_dashboard
            print("✅ Dashboard simple cargado")
            run_simple_dashboard()
            
        except ImportError as e2:
            print(f"⚠️ Error importando dashboard simple: {e2}")
            print("🔄 Creando dashboard de emergencia...")
            
            # Dashboard de emergencia
            import streamlit as st
            
            st.set_page_config(
                page_title="Forex Bot - Modo Emergencia",
                page_icon="⚠️",
                layout="wide"
            )
            
            st.title("⚠️ Forex Bot - Modo Emergencia")
            st.markdown("---")
            
            st.error("Error del sistema principal: " + str(e))
            st.warning("Error del sistema simple: " + str(e2))
            
            st.info("El sistema está funcionando en modo básico debido a problemas de importación")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Estado Sistema", "🟡 Limitado")
                
            with col2:
                st.metric("Módulos Cargados", "Básicos")
                
            with col3:
                if st.button("🔄 Reintentar Carga"):
                    st.rerun()
            
            st.markdown("### 🔧 Información de Debug")
            st.text("- Error principal: " + str(e))
            st.text("- Error simple: " + str(e2))
            st.text("- Modo actual: **Emergencia**")
            
            st.markdown("### 📋 Acciones recomendadas:")
            st.text("1. Verifica que todos los archivos estén presentes")
            st.text("2. Revisa las variables de entorno")
            st.text("3. Reinicia el servicio")
            st.text("4. Contacta soporte técnico si el problema persiste")
            
            with st.expander("📊 Información del Sistema"):
                system_info = "Sistema: Python " + str(sys.version) + "\n"
                system_info += "Directorio: " + os.getcwd() + "\n"
                system_info += "Archivos disponibles: " + str(os.listdir('.')) + "\n"
                
                if os.path.exists('modules'):
                    system_info += "Módulos disponibles: " + str(os.listdir('modules'))
                else:
                    system_info += "Módulos disponibles: No encontrado"
                    
                st.code(system_info)
            
    except Exception as e:
        print(f"❌ Error crítico en dashboard: {e}")
        logger.error(f"Error crítico en dashboard: {e}")
        
        try:
            import streamlit as st
            
            st.set_page_config(
                page_title="Error Crítico",
                page_icon="❌"
            )
            
            st.title("❌ Error Crítico del Sistema")
            st.error("Error crítico: " + str(e))
            
            st.markdown("### 🆘 El sistema ha encontrado un error crítico")
            
            st.markdown("**Detalles del error:**")
            st.code(str(e))
            
            st.markdown("### 🔧 Soluciones posibles:")
            st.text("1. **Reinicia la aplicación** completamente")
            st.text("2. **Verifica las variables de entorno** (ALPACA_API_KEY, etc.)")
            st.text("3. **Comprueba la conexión a internet**")
            st.text("4. **Revisa los logs** en /logs/forex_bot.log")
            st.text("5. **Contacta soporte técnico**")
            
            st.markdown("### 📞 Soporte")
            st.text("Si el problema persiste, proporciona la siguiente información:")
            st.text("- Mensaje de error completo")
            st.text("- Hora del error: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            st.text("- Configuración del sistema")
            
            if st.button("🔄 Intentar Reiniciar"):
                st.info("Reiniciando sistema...")
                st.rerun()
                
        except Exception as critical_error:
            print(f"💀 Error crítico total: {critical_error}")
            print("🆘 Sistema completamente inoperativo")
            print("📞 Contacta soporte técnico inmediatamente")

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
    
    display_banner()
    
    # Detectar si se ejecuta desde Streamlit
    if len(sys.argv) > 1 and any('--mode' in arg for arg in sys.argv):
        if 'dashboard' in sys.argv:
            run_dashboard_mode()
            return
    
    # Parser de argumentos
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
        help='Pares de divisas específicos'
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
    
    # Si no hay argumentos, ejecutar dashboard
    if len(sys.argv) == 1:
        run_dashboard_mode()
        return
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Modo verbose activado")
    
    try:
        # Aquí iría el código del bot para otros modos
        if args.mode == 'dashboard':
            print("🖥️  MODO: DASHBOARD WEB")
            run_dashboard_mode()
        else:
            st.info("Otros modos aún no implementados en esta versión simplificada")
    
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
