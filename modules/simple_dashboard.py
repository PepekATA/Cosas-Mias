import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

def run_simple_dashboard():
    """Dashboard simple y funcional"""
    
    # Configuración de la página
    st.set_page_config(
        page_title="🤖 Forex Prediction Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personalizado
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .prediction-up {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    .prediction-down {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Bot de Predicción de Divisas</h1>
        <h3>Sistema Inteligente de Trading con IA</h3>
        <p>📊 Dashboard en tiempo real | 🔮 Predicciones avanzadas | ⚡ Análisis instantáneo</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    create_sidebar()
    
    # Contenido principal
    main_content()

def create_sidebar():
    """Crear sidebar con controles"""
    st.sidebar.markdown("## ⚙️ Panel de Control")
    
    # Selección de pares
    st.sidebar.markdown("### 💱 Pares de Divisas")
    pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF', 'USD/CAD']
    selected_pairs = st.sidebar.multiselect(
        "Selecciona pares:",
        options=pairs,
        default=['EUR/USD', 'GBP/USD', 'USD/JPY'],
        help="Elige los pares de divisas para analizar"
    )
    
    # Intervalo de tiempo
    st.sidebar.markdown("### ⏰ Intervalo de Predicción")
    intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    selected_interval = st.sidebar.selectbox(
        "Intervalo:",
        options=intervals,
        index=1,
        help="Selecciona el horizonte temporal"
    )
    
    # Configuración de trading
    st.sidebar.markdown("### 🎯 Configuración")
    confidence_threshold = st.sidebar.slider(
        "Umbral de confianza:",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Confianza mínima para mostrar predicciones"
    )
    
    # Guardar en session state
    st.session_state.selected_pairs = selected_pairs
    st.session_state.selected_interval = selected_interval
    st.session_state.confidence_threshold = confidence_threshold
    
    # Botones de acción
    st.sidebar.markdown("### 🚀 Acciones")
    
    if st.sidebar.button("🎯 Generar Predicciones", type="primary"):
        generate_demo_predictions()
    
    if st.sidebar.button("📊 Actualizar Datos"):
        st.cache_data.clear()
        st.success("Cache actualizado")
        st.rerun()
    
    if st.sidebar.button("🧹 Limpiar Todo"):
        clear_session_state()
        st.rerun()
    
    # Estado del sistema
    st.sidebar.markdown("### 📊 Estado del Sistema")
    st.sidebar.success("✅ Sistema Operativo")
    st.sidebar.info(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    if st.session_state.get('last_update'):
        st.sidebar.text(f"Última actualización: {st.session_state.last_update}")

def main_content():
    """Contenido principal del dashboard"""
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Predicciones Actuales",
        "📊 Análisis de Mercado", 
        "🎯 Rendimiento",
        "⚙️ Configuración"
    ])
    
    with tab1:
        show_predictions_tab()
    
    with tab2:
        show_market_analysis_tab()
    
    with tab3:
        show_performance_tab()
    
    with tab4:
        show_settings_tab()

def show_predictions_tab():
    """Tab de predicciones"""
    st.markdown("## 📈 Predicciones en Tiempo Real")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Métricas principales
    with col1:
        st.metric("🎯 Predicciones Activas", 
                 st.session_state.get('total_predictions', 0),
                 delta="+2 hoy")
    
    with col2:
        st.metric("📊 Precisión Promedio", 
                 f"{st.session_state.get('avg_accuracy', 0.75):.1%}",
                 delta="+3.2%")
    
    with col3:
        st.metric("💰 Pares Monitoreados", 
                 len(st.session_state.get('selected_pairs', [])),
                 delta=None)
    
    with col4:
        st.metric("🔄 Estado", 
                 "🟢 Activo",
                 delta="Online")
    
    # Mostrar predicciones si existen
    if st.session_state.get('predictions'):
        st.markdown("### 🔮 Predicciones Generadas")
        display_predictions(st.session_state.predictions)
    else:
        # Card de instrucciones
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin: 2rem 0;">
            <h3>🎯 Comienza a generar predicciones</h3>
            <p style="font-size: 1.1em; color: #6c757d;">
                Usa el botón <strong>"🎯 Generar Predicciones"</strong> en el panel lateral para comenzar
            </p>
            <p>📊 Selecciona tus pares favoritos y configura el intervalo de tiempo</p>
        </div>
        """, unsafe_allow_html=True)

def display_predictions(predictions):
    """Mostrar predicciones en formato de cards"""
    
    for i, pred in enumerate(predictions):
        symbol = pred['symbol']
        direction = pred['direction']
        confidence = pred['confidence']
        price = pred['current_price']
        target = pred['target_price']
        
        # Determinar estilo
        card_class = "prediction-up" if direction == "UP" else "prediction-down"
        emoji = "🟢⬆️" if direction == "UP" else "🔴⬇️"
        
        # Calcular cambio esperado
        change_pct = ((target - price) / price) * 100 if price > 0 else 0
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <h4>{emoji} {symbol}</h4>
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <div>
                        <strong>Dirección:</strong> {direction}<br>
                        <strong>Confianza:</strong> {confidence:.1%}<br>
                        <strong>Precio Actual:</strong> {price:.5f}
                    </div>
                    <div>
                        <strong>Precio Objetivo:</strong> {target:.5f}<br>
                        <strong>Cambio Esperado:</strong> {change_pct:+.2f}%<br>
                        <strong>Duración:</strong> {pred.get('duration', 15)} min
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.7); padding: 0.5rem; border-radius: 5px;">
                    <div style="background: {'#28a745' if direction == 'UP' else '#dc3545'}; width: {confidence*100}%; height: 8px; border-radius: 4px;"></div>
                    <small>Nivel de confianza: {confidence:.1%}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Mini gráfico
            create_mini_chart(symbol, price, target, direction)

def create_mini_chart(symbol, current, target, direction):
    """Crear mini gráfico de predicción"""
    
    # Datos simulados para el gráfico
    x_data = list(range(10))
    y_data = np.random.randn(10).cumsum() + current
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name=symbol,
        line=dict(color='#28a745' if direction == 'UP' else '#dc3545', width=2)
    ))
    
    # Línea objetivo
    fig.add_hline(
        y=target,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Objetivo: {target:.5f}"
    )
    
    fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickformat='.5f')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_market_analysis_tab():
    """Tab de análisis de mercado"""
    st.markdown("## 📊 Análisis de Mercado")
    
    pairs = st.session_state.get('selected_pairs', ['EUR/USD', 'GBP/USD'])
    
    if pairs:
        selected_pair = st.selectbox("Seleccionar par para análisis:", pairs)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 📈 Gráfico de {selected_pair}")
            create_market_chart(selected_pair)
        
        with col2:
            st.markdown("### 🔍 Indicadores Técnicos")
            show_technical_indicators(selected_pair)
    else:
        st.warning("Selecciona al menos un par de divisas en la configuración")

def create_market_chart(pair):
    """Crear gráfico de mercado"""
    
    # Generar datos simulados
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    prices = np.random.randn(100).cumsum() + 1.1000
    
    # Crear candlestick simulado
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(100) * 0.001,
        'high': prices + abs(np.random.randn(100)) * 0.002,
        'low': prices - abs(np.random.randn(100)) * 0.002,
        'close': prices
    })
    
    fig = go.Figure(data=go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=pair
    ))
    
    # Agregar media móvil
    ma_20 = df['close'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=ma_20,
        mode='lines',
        name='MA 20',
        line=dict(color='orange', width=1)
    ))
    
    fig.update_layout(
        title=f"Gráfico de Velas - {pair}",
        yaxis_title="Precio",
        height=400,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_technical_indicators(pair):
    """Mostrar indicadores técnicos"""
    
    # Valores simulados
    indicators = {
        "RSI": np.random.uniform(30, 70),
        "MACD": np.random.uniform(-0.001, 0.001),
        "MA 20": np.random.uniform(1.0990, 1.1010),
        "MA 50": np.random.uniform(1.0980, 1.1020),
        "Bollinger %": np.random.uniform(0.2, 0.8),
        "ATR": np.random.uniform(0.001, 0.003)
    }
    
    for indicator, value in indicators.items():
        if indicator == "RSI":
            if value < 30:
                status = "🟢 Sobrevendido"
                delta = "Oportunidad"
            elif value > 70:
                status = "🔴 Sobrecomprado"
                delta = "Precaución"
            else:
                status = "🟡 Neutral"
                delta = None
            st.metric(indicator, f"{value:.1f}", delta=status)
        
        elif indicator == "MACD":
            st.metric(indicator, f"{value:.5f}", 
                     delta="Alcista" if value > 0 else "Bajista")
        
        elif "MA" in indicator:
            st.metric(indicator, f"{value:.4f}")
        
        elif indicator == "Bollinger %":
            st.metric(indicator, f"{value:.1%}")
        
        else:
            st.metric(indicator, f"{value:.4f}")

def show_performance_tab():
    """Tab de rendimiento"""
    st.markdown("## 🎯 Análisis de Rendimiento")
    
    col1, col2, col3 = st.columns(3)
    
    # Métricas de rendimiento simuladas
    with col1:
        st.metric("📊 Precisión General", "78.5%", delta="+2.3%")
        st.metric("🎯 Predicciones Totales", "247", delta="+15")
    
    with col2:
        st.metric("✅ Predicciones Correctas", "194", delta="+12")
        st.metric("📈 Ganancia Promedio", "0.85%", delta="+0.12%")
    
    with col3:
        st.metric("⏱️ Tiempo Promedio", "12 min", delta="-2 min")
        st.metric("🔥 Racha Actual", "7 días", delta="+1 día")
    
    # Gráfico de rendimiento
    st.markdown("### 📈 Historial de Precisión")
    
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    accuracy = np.random.uniform(0.6, 0.9, 30)
    
    fig = px.line(
        x=dates,
        y=accuracy,
        title="Evolución de la Precisión (Últimos 30 días)",
        labels={'x': 'Fecha', 'y': 'Precisión'}
    )
    
    fig.update_traces(line=dict(color='#28a745', width=3))
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de rendimiento por par
    st.markdown("### 📋 Rendimiento por Par de Divisas")
    
    pairs_data = {
        'Par': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
        'Predicciones': [45, 38, 42, 35],
        'Correctas': [36, 29, 31, 26],
        'Precisión': ['80.0%', '76.3%', '73.8%', '74.3%'],
        'Ganancia Avg': ['0.92%', '0.88%', '0.73%', '0.81%']
    }
    
    df_performance = pd.DataFrame(pairs_data)
    st.dataframe(df_performance, use_container_width=True)

def show_settings_tab():
    """Tab de configuración"""
    st.markdown("## ⚙️ Configuración del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Configuración de Trading")
        
        risk_level = st.selectbox(
            "Nivel de riesgo:",
            ["Conservador", "Moderado", "Agresivo"],
            index=1
        )
        
        max_positions = st.number_input(
            "Máximo de posiciones simultáneas:",
            min_value=1,
            max_value=10,
            value=3
        )
        
        auto_update = st.checkbox("Actualización automática", value=True)
        
        notifications = st.checkbox("Notificaciones push", value=False)
    
    with col2:
        st.markdown("### 📊 Estado del Sistema")
        
        system_status = {
            "🤖 Predictor": "✅ Operativo",
            "📊 Data Collector": "✅ Conectado",
            "🔮 ML Models": "✅ Entrenados",
            "📡 API Status": "✅ Online",
            "💾 Storage": "✅ Disponible"
        }
        
        for component, status in system_status.items():
            st.text(f"{component}: {status}")
        
        st.markdown("### 🧹 Mantenimiento")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            if st.button("🗑️ Limpiar Cache"):
                st.cache_data.clear()
                st.success("Cache limpiado")
        
        with col2b:
            if st.button("🔄 Reiniciar Modelos"):
                st.info("Reiniciando modelos...")
    
    # Información del sistema
    st.markdown("### 📋 Información del Sistema")
    
    system_info = f"""
    - **Versión:** Forex Predictor v2.0
    - **Última actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - **Modelos activos:** LSTM, Random Forest, XGBoost
    - **Pares monitoreados:** {len(st.session_state.get('selected_pairs', []))}
    - **Predicciones hoy:** {st.session_state.get('daily_predictions', 0)}
    """
    
    st.markdown(system_info)

def generate_demo_predictions():
    """Generar predicciones de demostración"""
    
    pairs = st.session_state.get('selected_pairs', ['EUR/USD', 'GBP/USD'])
    
    if not pairs:
        st.sidebar.error("Selecciona al menos un par de divisas")
        return
    
    # Simular generación de predicciones
    predictions = []
    
    for pair in pairs:
        prediction = {
            'symbol': pair,
            'direction': np.random.choice(['UP', 'DOWN']),
            'confidence': np.random.uniform(0.65, 0.92),
            'current_price': np.random.uniform(1.0500, 1.2000),
            'target_price': 0,
            'duration': np.random.randint(5, 45),
            'timestamp': datetime.now()
        }
        
        # Calcular precio objetivo
        multiplier = 1.002 if prediction['direction'] == 'UP' else 0.998
        prediction['target_price'] = prediction['current_price'] * multiplier
        
        predictions.append(prediction)
    
    # Guardar en session state
    st.session_state.predictions = predictions
    st.session_state.total_predictions = len(predictions)
    st.session_state.last_update = datetime.now().strftime('%H:%M:%S')
    
    st.sidebar.success(f"✅ {len(predictions)} predicciones generadas")
    st.rerun()

def clear_session_state():
    """Limpiar datos de la sesión"""
    keys_to_clear = ['predictions', 'total_predictions', 'last_update']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Datos limpiados")

if __name__ == "__main__":
    run_simple_dashboard()
