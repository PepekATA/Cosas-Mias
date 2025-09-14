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

# ============== AQUÍ VA EL CÓDIGO ACTUALIZADO ==============
def run_dashboard_mode():
    """Ejecuta el dashboard web de Streamlit"""
    print("🖥️  INICIANDO DASHBOARD WEB")
    print("="*40)
    print("🌐 Abriendo interfaz web...")
    print("📱 Accede desde tu navegador")
    print("⏹️  Presiona Ctrl+C para detener")
    print("="*40)
    
    try:
        # Intentar importar y ejecutar dashboard principal
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
            
            # Dashboard de emergencia inline
            import streamlit as st
            
            st.set_page_config(
                page_title="Forex Bot - Modo Emergencia",
                page_icon="⚠️",
                layout="wide"
            )
            
            st.title("⚠️ Forex Bot - Modo Emergencia")
            st.markdown("---")
            
            st.error(f"Error del sistema principal: {e}")
            st.warning(f"Error del sistema simple: {e2}")
            
            st.info("El sistema está funcionando en modo básico debido a problemas de importación")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Estado Sistema", "🟡 Limitado")
                
            with col2:
                st.metric("Módulos Cargados", "Básicos")
                
            with col3:
                if st.button("🔄 Reintentar Carga"):
                    st.rerun()
            
            st.markdown("""
            ### 🔧 Información de Debug
            - Error principal: `{}`
            - Error simple: `{}`
            - Modo actual: **Emergencia**
            
            ### 📋 Acciones recomendadas:
            1. Verifica que todos los archivos estén presentes
            2. Revisa las variables de entorno
            3. Reinicia el servicio
            4. Contacta soporte técnico si el problema persiste
            """.format(e, e2))
            
            # Mostrar información del sistema
            with st.expander("📊 Información del Sistema"):
                st.code(f"""
Sistema: Python {sys.version}
Directorio: {os.getcwd()}
Archivos disponibles: {os.listdir('.')}
Módulos disponibles: {os.listdir('modules') if os.path.exists('modules') else 'No encontrado'}
                """)
            
    except Exception as e:
        print(f"❌ Error crítico en dashboard: {e}")
        logger.error(f"Error crítico en dashboard: {e}")
        
        # Dashboard de error crítico
        try:
            import streamlit as st
            
            st.set_page_config(
                page_title="Error Crítico",
                page_icon="❌"
            )
            
            st.title("❌ Error Crítico del Sistema")
            st.error(f"Error crítico: {e}")
            
            st.markdown("""
            ### 🆘 El sistema ha encontrado un error crítico
            
            **Detalles del error:**
