#!/usr/bin/env python3
"""
Principal.py - App Streamlit modular para Bot de Predicción de Divisas
Lista para producción
"""
import os
import sys
import time
import logging
import warnings
import streamlit as st
from datetime import datetime, timedelta

# Ignorar warnings
warnings.filterwarnings('ignore')

# Agregar directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos internos
from modules.almacenamiento import get_alpaca_client
from modules.predictor import generar_predicciones
from modules.tablero_simple import mostrar_dashboard, mostrar_charts
from modules.panel import mostrar_panel_control

# Crear directorios necesarios
for dir_path in ['data/historical', 'data/models', 'logs']:
    os.makedirs(dir_path, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/forex_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Inicializar Alpaca Client
client = get_alpaca_client()

# --- FUNCIONES AUXILIARES ---
def init_session_state():
    """Inicializa las variables de sesión necesarias"""
    if "selected_pairs" not in st.session_state:
        st.session_state.selected_pairs = ["EUR/USD", "GBP/USD"]
    if "interval" not in st.session_state:
        st.session_state.interval = "5m"

# --- APP STREAMLIT ---
def run_streamlit_app():
    """Función principal de Streamlit"""
    st.set_page_config(
        page_title="🤖 Forex Bot - Predicciones IA",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inicializar session state
    init_session_state()

    # CSS personalizado
    st.markdown("""
    <style>
        .main > div { padding: 1rem; }
        .stApp { background-color: #f5f5f5; }
        .metric-container { background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🤖 Bot de Predicción de Divisas")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        mostrar_panel_control()

        # Configuración de pares e intervalos
        pares_de_divisas = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP"
        ]
        st.session_state.selected_pairs = st.multiselect(
            "Pares de Divisas:",
            options=pares_de_divisas,
            default=st.session_state.selected_pairs
        )

        st.session_state.interval = st.selectbox(
            "Intervalo:",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "1h", "4h", "1d"].index(st.session_state.interval)
        )

        # Botones
        if st.button("🚀 Generar Predicciones"):
            generar_predicciones(st.session_state.selected_pairs, st.session_state.interval)

    # Contenido principal con tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predicciones", "📈 Gráficos", "⚙️ Configuración"])

    with tab1:
        mostrar_dashboard()
    with tab2:
        st.subheader("🎯 Predicciones de IA")
        st.info("Predicciones se muestran aquí...")
    with tab3:
        mostrar_charts()
    with tab4:
        st.subheader("⚙️ Configuración Avanzada")
        st.info("Aquí puedes configurar parámetros de modelos y API keys")

# --- PUNTO DE ENTRADA ---
def main():
    """Inicia la aplicación"""
    if 'streamlit' in sys.modules:
        run_streamlit_app()
    else:
        print("🚀 Iniciando aplicación Streamlit...")
        os.system(f"{sys.executable} -m streamlit run {__file__} --server.port=8501")

if __name__ == "__main__":
    main()
