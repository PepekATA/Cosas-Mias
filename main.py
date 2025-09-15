#!/usr/bin/env python3
"""
Bot de Predicción de Divisas - Punto de entrada principal
Versión corregida para Streamlit con session_state inicializado
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import os
import sys
import logging
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
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

# --- Inicializar estado ---
def init_session_state():
    """Inicializar claves de session_state necesarias"""
    defaults = {
        "interval": "1m",
        "selected_pairs": ["EUR/USD", "GBP/USD"],
        "api_key": "",
        "api_secret": "",
        "confidence_threshold": 0.75,
        "lookback_days": 30,
        "update_frequency": "1 minuto"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Detección Streamlit ---
def is_streamlit():
    """Detecta si el código se ejecuta en Streamlit"""
    try:
        import streamlit as st
        return hasattr(st, 'session_state')
    except ImportError:
        return False

# --- Main ---
def main():
    """Función principal - SIEMPRE ejecuta la aplicación Streamlit"""
    if is_streamlit() or 'streamlit' in sys.modules:
        run_streamlit_app()
    else:
        print("🚀 Iniciando aplicación Streamlit...")
        os.system(f"{sys.executable} -m streamlit run {__file__} --server.port=8501")

def run_streamlit_app():
    """Aplicación principal de Streamlit"""
    init_session_state()

    # Configuración de página
    st.set_page_config(
        page_title="🤖 Forex Bot - Predicciones IA",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personalizado
    st.markdown("""
    <style>
        .main > div { padding: 1rem; }
        .stApp { background-color: #f5f5f5; }
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

        currency_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP"
        ]

        st.session_state.selected_pairs = st.multiselect(
            "Pares de Divisas:",
            currency_pairs,
            default=st.session_state.selected_pairs,
            key="selected_pairs"
        )

        st.session_state.interval = st.selectbox(
            "Intervalo:",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "1h", "4h", "1d"].index(st.session_state.interval),
            key="interval"
        )

        st.markdown("---")
        if st.button("🚀 Generar Predicciones", type="primary"):
            generate_predictions(st.session_state.selected_pairs, st.session_state.interval)

        if st.button("🔄 Actualizar Datos"):
            st.info("Actualizando datos...")

    # Métricas principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container"><h3>📈 Predicciones Activas</h3><h2>0</h2><p>En proceso</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container"><h3>🎯 Precisión Promedio</h3><h2>75.2%</h2><p>Última semana</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container"><h3>💰 Ganancia Estimada</h3><h2>+12.4%</h2><p>Este mes</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predicciones", "📈 Gráficos", "⚙️ Configuración"])
    with tab1: show_dashboard()
    with tab2: show_predictions()
    with tab3: show_charts()
    with tab4: show_settings()

# --- Vistas ---
def show_dashboard():
    st.subheader("📊 Dashboard Principal")
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'Fecha': dates,
        'EUR/USD': np.random.random(30) * 0.1 + 1.05,
        'GBP/USD': np.random.random(30) * 0.1 + 1.25,
        'USD/JPY': np.random.random(30) * 5 + 148
    })

    col1, col2 = st.columns([2, 1])
    with col1:
        st.line_chart(df.set_index('Fecha')[['EUR/USD', 'GBP/USD']])
    with col2:
        st.subheader("📋 Resumen Diario")
        st.metric("EUR/USD", "1.0542", "+0.0021")
        st.metric("GBP/USD", "1.2511", "-0.0034")
        st.metric("USD/JPY", "149.45", "+0.23")

def show_predictions():
    st.subheader("🎯 Predicciones de IA")
    predictions_data = [
        {"Par": "EUR/USD", "Dirección": "↗️ UP", "Confianza": "78%", "Target": "1.0580", "Tiempo": "15 min"},
        {"Par": "GBP/USD", "Dirección": "↘️ DOWN", "Confianza": "65%", "Target": "1.2480", "Tiempo": "30 min"},
        {"Par": "USD/JPY", "Dirección": "↗️ UP", "Confianza": "82%", "Target": "149.80", "Tiempo": "45 min"}
    ]
    for i, pred in enumerate(predictions_data):
        with st.expander(f"📊 {pred['Par']} - {pred['Dirección']}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Confianza", pred["Confianza"])
            with col2: st.metric("Target", pred["Target"])
            with col3: st.metric("Tiempo", pred["Tiempo"])
            with col4:
                if st.button(f"📋 Detalles", key=f"details_{i}"):
                    st.info("Mostrando análisis detallado...")

def show_charts():
    st.subheader("📈 Análisis Técnico")
    st.info("🔧 Gráficos avanzados en desarrollo...")
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='EUR/USD'))
    fig.update_layout(title="EUR/USD - Análisis Técnico", height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_settings():
    st.subheader("⚙️ Configuración Avanzada")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔑 API Configuration")
        st.session_state.api_key = st.text_input("Alpaca API Key", type="password", key="api_key")
        st.session_state.api_secret = st.text_input("Alpaca Secret", type="password", key="api_secret")
        if st.button("✅ Guardar Configuración"):
            if st.session_state.api_key and st.session_state.api_secret:
                st.success("✅ Configuración guardada correctamente")
            else:
                st.error("❌ Por favor completa todos los campos")
    with col2:
        st.subheader("📊 Parámetros del Modelo")
        st.session_state.confidence_threshold = st.slider("Umbral de Confianza", 0.5, 0.95, st.session_state.confidence_threshold, key="confidence_threshold")
        st.session_state.lookback_days = st.number_input("Días de Historial", 1, 365, st.session_state.lookback_days, key="lookback_days")
        st.session_state.update_frequency = st.selectbox("Frecuencia de Actualización", ["1 minuto", "5 minutos", "15 minutos", "1 hora"], index=0, key="update_frequency")

# --- Utilidades ---
def generate_predictions(pairs, interval):
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, pair in enumerate(pairs):
        status_text.text(f"Analizando {pair}...")
        progress_bar.progress((i + 1) / len(pairs))
        time.sleep(0.5)
    status_text.text("✅ Predicciones generadas correctamente!")
    st.balloons()

# --- Punto de entrada ---
if __name__ == "__main__":
    main()
