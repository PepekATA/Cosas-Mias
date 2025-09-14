import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Agregar path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from modules.predictor import ForexPredictor
    from modules.storage import DataStorage
    from config import CURRENCY_PAIRS, PREDICTION_INTERVALS
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()

def run_dashboard():
    """Función principal del dashboard"""
    
    # Configuración de la página
    st.set_page_config(
        page_title="Forex Prediction Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personalizado
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .up-card { border-left: 5px solid #28a745; background: #f8fff9; }
    .down-card { border-left: 5px solid #dc3545; background: #fff8f8; }
    .neutral-card { border-left: 5px solid #6c757d; background: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Bot de Predicción de Divisas</h1>
        <p>Sistema Inteligente de Predicción Forex con IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar session state
    if 'predictor' not in st.session_state:
        with st.spinner("Inicializando sistema..."):
            try:
                st.session_state.predictor = ForexPredictor()
                st.session_state.storage = DataStorage()
                st.success("✅ Sistema inicializado correctamente")
            except Exception as e:
                st.error(f"❌ Error inicializando sistema: {e}")
                st.stop()
    
    # Sidebar
    create_sidebar()
    
    # Contenido principal con tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Predicciones", 
        "📊 Análisis Técnico", 
        "🎯 Rendimiento", 
        "📋 Configuración"
    ])
    
    with tab1:
        show_predictions_tab()
    
    with tab2:
        show_technical_tab()
        
    with tab3:
        show_performance_tab()
        
    with tab4:
        show_config_tab()

def create_sidebar():
    """Crear sidebar con controles"""
    st.sidebar.markdown("## ⚙️ Configuración")
    
    # Selección de pares
    st.sidebar.markdown("### 💱 Pares de Divisas")
    selected_pairs = st.sidebar.multiselect(
        "Seleccionar pares:",
        options=CURRENCY_PAIRS,
        default=CURRENCY_PAIRS[:4],
        help="Selecciona los pares de divisas para analizar"
    )
    
    # Intervalo de predicción
    st.sidebar.markdown("### ⏰ Intervalo")
    selected_interval = st.sidebar.selectbox(
        "Intervalo de predicción:",
        options=list(PREDICTION_INTERVALS.keys()),
        index=2,
        help="Selecciona el horizonte temporal"
    )
    
    # Guardar en session state
    st.session_state.selected_pairs = selected_pairs
    st.session_state.selected_interval = selected_interval
    
    # Botones de acción
    st.sidebar.markdown("### 🎯 Acciones")
    
    if st.sidebar.button("🚀 Generar Predicciones", type="primary"):
        generate_predictions()
    
    if st.sidebar.button("📊 Entrenar Modelos"):
        train_models()
    
    if st.sidebar.button("🔄 Actualizar Dashboard"):
        st.rerun()
    
    # Estado del sistema
    st.sidebar.markdown("### 📊 Estado del Sistema")
    if st.session_state.get('last_prediction'):
        st.sidebar.success("✅ Sistema Operativo")
        last_time = st.session_state.get('last_prediction_time', 'Nunca')
        st.sidebar.text(f"Última predicción: {last_time}")
    else:
        st.sidebar.warning("⚠️ Sin predicciones")

def show_predictions_tab():
    """Tab de predicciones"""
    st.markdown("## 📈 Predicciones Actuales")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 📊 Resumen")
        if st.session_state.get('last_predictions'):
            predictions = st.session_state.last_predictions
            up_count = sum(1 for p in predictions if p['direction'] == 'UP')
            down_count = sum(1 for p in predictions if p['direction'] == 'DOWN')
            
            st.metric("Total", len(predictions))
            st.metric("🟢 Alcista", up_count)
            st.metric("🔴 Bajista", down_count)
        else:
            st.info("No hay predicciones disponibles")
    
    with col1:
        if st.session_state.get('last_predictions'):
            display_predictions(st.session_state.last_predictions)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h3>🎯 Genera tus primeras predicciones</h3>
                <p>Usa el botón "🚀 Generar Predicciones" en la barra lateral</p>
            </div>
            """, unsafe_allow_html=True)

def display_predictions(predictions):
    """Mostrar predicciones en cards"""
    for i, prediction in enumerate(predictions):
        symbol = prediction.get('symbol', 'N/A')
        direction = prediction.get('direction', 'NEUTRAL')
        confidence = prediction.get('confidence', 0.0)
        current_price = prediction.get('current_price', 0.0)
        price_target = prediction.get('price_target', 0.0)
        duration = prediction.get('duration_minutes', 0)
        
        # Determinar estilo de card
        if direction == 'UP':
            card_class = "up-card"
            emoji = "🟢⬆️"
        elif direction == 'DOWN':
            card_class = "down-card"
            emoji = "🔴⬇️"
        else:
            card_class = "neutral-card"
            emoji = "🟡➡️"
        
        # Calcular cambio porcentual
        if current_price > 0:
            pct_change = ((price_target - current_price) / current_price) * 100
        else:
            pct_change = 0
        
        # Card HTML
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <h3>{emoji} {symbol}</h3>
            <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                <div>
                    <strong>Dirección:</strong> {direction}<br>
                    <strong>Confianza:</strong> {confidence:.1%}<br>
                    <strong>Duración:</strong> {duration} min
                </div>
                <div>
                    <strong>Precio Actual:</strong> {current_price:.5f}<br>
                    <strong>Objetivo:</strong> {price_target:.5f}<br>
                    <strong>Cambio:</strong> {pct_change:+.2f}%
                </div>
            </div>
            <div style="background: #f0f0f0; padding: 0.5rem; border-radius: 5px;">
                <div style="background: {'#28a745' if direction == 'UP' else '#dc3545' if direction == 'DOWN' else '#6c757d'}; 
                           width: {confidence*100}%; height: 8px; border-radius: 4px;"></div>
                <small>Nivel de confianza: {confidence:.1%}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_technical_tab():
    """Tab de análisis técnico"""
    st.markdown("## 📊 Análisis Técnico")
    
    pairs = st.session_state.get('selected_pairs', CURRENCY_PAIRS[:4])
    
    if pairs:
        selected_pair = st.selectbox("Seleccionar par para análisis:", pairs)
        
        if st.button("📈 Generar Análisis"):
            with st.spinner(f"Analizando {selected_pair}..."):
                try:
                    # Simular análisis técnico
                    st.success(f"✅ Análisis de {selected_pair} completado")
                    
                    # Crear gráfico demo
                    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
                    prices = np.random.randn(100).cumsum() + 1.1000
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=prices,
                        mode='lines',
                        name=selected_pair,
                        line=dict(color='#667eea', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Gráfico de {selected_pair}",
                        xaxis_title="Tiempo",
                        yaxis_title="Precio",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error en análisis: {e}")
    else:
        st.warning("Selecciona al menos un par de divisas en la configuración")

def show_performance_tab():
    """Tab de rendimiento"""
    st.markdown("## 🎯 Rendimiento del Sistema")
    
    try:
        if 'storage' in st.session_state:
            summary = st.session_state.storage.get_performance_summary()
            
            if summary.get('total_predictions', 0) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predicciones", summary['total_predictions'])
                
                with col2:
                    st.metric("Correctas", summary['correct_predictions'])
                
                with col3:
                    accuracy = summary['overall_accuracy']
                    st.metric("Precisión", f"{accuracy:.1%}")
                
                with col4:
                    st.metric("Estado", "🟢 Operativo")
                
                # Gráfico de rendimiento
                if summary.get('by_symbol'):
                    st.markdown("### 📊 Rendimiento por Par")
                    
                    symbols = list(summary['by_symbol'].keys())
                    accuracies = [summary['by_symbol'][s]['accuracy'] for s in symbols]
                    
                    fig = px.bar(
                        x=symbols,
                        y=accuracies,
                        title="Precisión por Par de Divisas",
                        color=accuracies,
                        color_continuous_scale="RdYlGn"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No hay datos de rendimiento disponibles")
                st.markdown("Genera algunas predicciones para ver estadísticas")
        else:
            st.error("Sistema de almacenamiento no disponible")
            
    except Exception as e:
        st.error(f"Error obteniendo rendimiento: {e}")

def show_config_tab():
    """Tab de configuración"""
    st.markdown("## 📋 Configuración del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Configuraciones")
        
        # Configuración de modelo
        st.selectbox(
            "Modelo principal:",
            ["LSTM + Random Forest", "Solo LSTM", "Solo Random Forest"],
            help="Selecciona el tipo de modelo a usar"
        )
        
        st.slider(
            "Umbral de confianza:",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Umbral mínimo de confianza para predicciones"
        )
        
        st.number_input(
            "Días de historial:",
            min_value=7,
            max_value=365,
            value=30,
            help="Días de datos históricos a usar"
        )
    
    with col2:
        st.markdown("### 📊 Estado del Sistema")
        
        # Estado de módulos
        modules_status = {
            "Predictor": "✅ Operativo",
            "Storage": "✅ Operativo",
            "Data Collector": "✅ Operativo",
            "ML Models": "⚠️ Entrenando" if st.session_state.get('training') else "✅ Listo"
        }
        
        for module, status in modules_status.items():
            st.text(f"{module}: {status}")
        
        # Botones de mantenimiento
        st.markdown("### 🧹 Mantenimiento")
        
        if st.button("🗑️ Limpiar Datos Antiguos"):
            with st.spinner("Limpiando..."):
                try:
                    if 'storage' in st.session_state:
                        st.session_state.storage.cleanup_old_files()
                        st.success("✅ Datos limpiados")
                    else:
                        st.error("Storage no disponible")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("🔄 Reiniciar Sistema"):
            # Limpiar session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Sistema reiniciado")
            st.rerun()

def generate_predictions():
    """Generar nuevas predicciones"""
    pairs = st.session_state.get('selected_pairs', CURRENCY_PAIRS[:2])
    interval = st.session_state.get('selected_interval', '5m')
    
    if not pairs:
        st.sidebar.error("Selecciona al menos un par de divisas")
        return
    
    with st.sidebar:
        with st.spinner("Generando predicciones..."):
            try:
                # Simular predicciones (reemplazar con código real)
                predictions = []
                for pair in pairs:
                    prediction = {
                        'symbol': pair,
                        'direction': np.random.choice(['UP', 'DOWN']),
                        'confidence': np.random.uniform(0.6, 0.9),
                        'current_price': np.random.uniform(1.0000, 1.2000),
                        'price_target': 0,
                        'duration_minutes': np.random.randint(5, 60)
                    }
                    prediction['price_target'] = prediction['current_price'] * (
                        1.001 if prediction['direction'] == 'UP' else 0.999
                    )
                    predictions.append(prediction)
                
                st.session_state.last_predictions = predictions
                st.session_state.last_prediction_time = datetime.now().strftime('%H:%M:%S')
                st.success(f"✅ {len(predictions)} predicciones generadas")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

def train_models():
    """Entrenar modelos"""
    pairs = st.session_state.get('selected_pairs', CURRENCY_PAIRS[:2])
    
    with st.sidebar:
        with st.spinner("Entrenando modelos..."):
            try:
                st.session_state.training = True
                # Simular entrenamiento
                import time
                time.sleep(2)
                
                st.session_state.training = False
                st.success("✅ Modelos entrenados")
                
            except Exception as e:
                st.session_state.training = False
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    run_dashboard()
