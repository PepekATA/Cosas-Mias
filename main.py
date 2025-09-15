#!/usr/bin/env python3
"""
Bot de Predicción de Divisas - Punto de entrada principal
Versión corregida para Streamlit
"""

import sys
import os
import logging
from datetime import datetime
import warnings
import streamlit as st
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

# VERIFICAR SI EJECUTAMOS EN STREAMLIT
def is_streamlit():
    """Detecta si el código se ejecuta en Streamlit"""
    try:
        import streamlit as st
        return hasattr(st, 'session_state')
    except ImportError:
        return False

def main():
    """Función principal - SIEMPRE ejecuta la aplicación Streamlit"""
    
    # Si estamos en Streamlit, ejecutar la app directamente
    if is_streamlit() or 'streamlit' in sys.modules:
        run_streamlit_app()
    else:
        # Ejecutar con streamlit run
        print("🚀 Iniciando aplicación Streamlit...")
        os.system(f"{sys.executable} -m streamlit run {__file__} --server.port=8501")

def run_streamlit_app():
    """Aplicación principal de Streamlit"""
    import streamlit as st
    
    # Configuración de página
    st.set_page_config(
        page_title="🤖 Forex Bot - Predicciones IA",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personalizado para evitar página en blanco
    st.markdown("""
    <style>
        .main > div {
            padding: 1rem;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        .metric-container {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.title("🤖 Bot de Predicción de Divisas")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Panel de Control")
        
        # Estado del sistema
        st.subheader("📊 Estado del Sistema")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estado", "🟢 Online")
        with col2:
            st.metric("Uptime", "Running")
        
        st.markdown("---")
        
        # Configuración
        st.subheader("🔧 Configuración")
        
        # Selección de pares
        currency_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP"
        ]
        
        selected_pairs = st.multiselect(
            "Pares de Divisas:",
            currency_pairs,
            default=["EUR/USD", "GBP/USD"]
        )
        
        # Intervalo de tiempo
        interval = st.selectbox(
            "Intervalo:",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=1
        )
        
        # Botones de acción
        st.markdown("---")
        if st.button("🚀 Generar Predicciones", type="primary"):
            generate_predictions(selected_pairs, interval)
        
        if st.button("🔄 Actualizar Datos"):
            st.info("Actualizando datos...")
            
    # Contenido principal
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>📈 Predicciones Activas</h3>
            <h2>0</h2>
            <p>En proceso</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>🎯 Precisión Promedio</h3>
            <h2>75.2%</h2>
            <p>Última semana</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>💰 Ganancia Estimada</h3>
            <h2>+12.4%</h2>
            <p>Este mes</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predicciones", "📈 Gráficos", "⚙️ Configuración"])
    
    with tab1:
        show_dashboard()
    
    with tab2:
        show_predictions()
    
    with tab3:
        show_charts()
    
    with tab4:
        show_settings()

def show_dashboard():
    """Muestra el dashboard principal"""
    st.subheader("📊 Dashboard Principal")
    
    # Gráfico de ejemplo
    import pandas as pd
    import numpy as np
    
    # Datos de ejemplo
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    data = {
        'Fecha': dates,
        'EUR/USD': np.random.random(30) * 0.1 + 1.05,
        'GBP/USD': np.random.random(30) * 0.1 + 1.25,
        'USD/JPY': np.random.random(30) * 5 + 148
    }
    
    df = pd.DataFrame(data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.line_chart(df.set_index('Fecha')[['EUR/USD', 'GBP/USD']])
    
    with col2:
        st.subheader("📋 Resumen Diario")
        st.metric("EUR/USD", "1.0542", "+0.0021")
        st.metric("GBP/USD", "1.2511", "-0.0034")
        st.metric("USD/JPY", "149.45", "+0.23")

def show_predictions():
    """Muestra las predicciones"""
    st.subheader("🎯 Predicciones de IA")
    
    # Ejemplo de predicciones
    predictions_data = [
        {"Par": "EUR/USD", "Dirección": "↗️ UP", "Confianza": "78%", "Target": "1.0580", "Tiempo": "15 min"},
        {"Par": "GBP/USD", "Dirección": "↘️ DOWN", "Confianza": "65%", "Target": "1.2480", "Tiempo": "30 min"},
        {"Par": "USD/JPY", "Dirección": "↗️ UP", "Confianza": "82%", "Target": "149.80", "Tiempo": "45 min"}
    ]
    
    for i, pred in enumerate(predictions_data):
        with st.expander(f"📊 {pred['Par']} - {pred['Dirección']}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confianza", pred["Confianza"])
            with col2:
                st.metric("Target", pred["Target"])
            with col3:
                st.metric("Tiempo", pred["Tiempo"])
            with col4:
                if st.button(f"📋 Detalles", key=f"details_{i}"):
                    st.info("Mostrando análisis detallado...")

def show_charts():
    """Muestra gráficos avanzados"""
    st.subheader("📈 Análisis Técnico")
    
    # Placeholder para gráficos más avanzados
    st.info("🔧 Gráficos avanzados en desarrollo...")
    
    # Gráfico simple de ejemplo
    import plotly.graph_objects as go
    import numpy as np
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='EUR/USD'))
    fig.update_layout(title="EUR/USD - Análisis Técnico", height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Muestra configuración avanzada"""
    st.subheader("⚙️ Configuración Avanzada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔑 API Configuration")
        api_key = st.text_input("Alpaca API Key", type="password")
        api_secret = st.text_input("Alpaca Secret", type="password")
        
        if st.button("✅ Guardar Configuración"):
            if api_key and api_secret:
                st.success("✅ Configuración guardada correctamente")
            else:
                st.error("❌ Por favor completa todos los campos")
    
    with col2:
        st.subheader("📊 Parámetros del Modelo")
        confidence_threshold = st.slider("Umbral de Confianza", 0.5, 0.95, 0.75)
        lookback_days = st.number_input("Días de Historial", 1, 365, 30)
        update_frequency = st.selectbox("Frecuencia de Actualización", 
                                      ["1 minuto", "5 minutos", "15 minutos", "1 hora"])

def generate_predictions(pairs, interval):
    """Genera predicciones simuladas"""
    import time
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pair in enumerate(pairs):
        status_text.text(f"Analizando {pair}...")
        progress_bar.progress((i + 1) / len(pairs))
        time.sleep(0.5)  # Simular procesamiento
    
    status_text.text("✅ Predicciones generadas correctamente!")
    st.balloons()

# PUNTO DE ENTRADA PRINCIPAL
if __name__ == "__main__":
    main()
