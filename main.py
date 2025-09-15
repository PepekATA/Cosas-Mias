#!/usr/bin/env python3
"""
🤖 Bot de Predicción de Divisas con IA
Optimizado para Streamlit Cloud (streamlit.io)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random

# ===== CONFIGURACIÓN DE PÁGINA =====
st.set_page_config(
    page_title="🤖 Forex Bot IA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/forex-bot',
        'Report a bug': "https://github.com/tu-usuario/forex-bot/issues",
        'About': "# Forex Bot v2.0\nBot de predicción de divisas con IA"
    }
)

# ===== CSS PERSONALIZADO =====
st.markdown("""
<style>
    /* Tema oscuro personalizado */
    .main > div {
        padding: 2rem 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    
    /* Animaciones */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ===== CONFIGURACIÓN INICIAL =====
@st.cache_data
def load_currency_pairs():
    """Carga los pares de divisas disponibles"""
    return {
        "Principales": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"],
        "Secundarios": ["AUD/USD", "USD/CAD", "NZD/USD", "USD/SEK"],
        "Exóticos": ["EUR/GBP", "EUR/JPY", "GBP/JPY", "CHF/JPY"],
        "Criptomonedas": ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD"]
    }

@st.cache_data
def generate_sample_data(pair, days=30):
    """Genera datos de muestra para gráficos"""
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    
    # Simular precio inicial basado en el par
    if "EUR/USD" in pair:
        base_price = 1.0500
        volatility = 0.002
    elif "GBP/USD" in pair:
        base_price = 1.2500
        volatility = 0.003
    elif "USD/JPY" in pair:
        base_price = 148.50
        volatility = 0.5
    else:
        base_price = 1.0000
        volatility = 0.002
    
    # Generar precio con tendencia y volatilidad
    prices = []
    price = base_price
    
    for i in range(len(dates)):
        # Añadir tendencia sutil
        trend = np.sin(i / 100) * 0.001
        # Añadir ruido aleatorio
        noise = np.random.normal(0, volatility)
        price += trend + noise
        prices.append(price)
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })

# ===== FUNCIONES DE PREDICCIÓN =====
def generate_prediction(pair):
    """Genera una predicción simulada"""
    directions = ["↗️ COMPRA", "↘️ VENTA", "➡️ MANTENER"]
    direction = random.choice(directions)
    confidence = random.uniform(0.65, 0.95)
    
    # Precio actual simulado
    if "EUR/USD" in pair:
        current_price = round(random.uniform(1.0400, 1.0600), 4)
    elif "GBP/USD" in pair:
        current_price = round(random.uniform(1.2400, 1.2600), 4)
    elif "USD/JPY" in pair:
        current_price = round(random.uniform(147.00, 150.00), 2)
    else:
        current_price = round(random.uniform(0.9000, 1.1000), 4)
    
    # Target price
    if "COMPRA" in direction:
        target_price = current_price * (1 + random.uniform(0.001, 0.01))
    elif "VENTA" in direction:
        target_price = current_price * (1 - random.uniform(0.001, 0.01))
    else:
        target_price = current_price * (1 + random.uniform(-0.002, 0.002))
    
    if "JPY" in pair:
        target_price = round(target_price, 2)
    else:
        target_price = round(target_price, 4)
    
    return {
        'pair': pair,
        'direction': direction,
        'confidence': confidence,
        'current_price': current_price,
        'target_price': target_price,
        'time_frame': random.choice(["15 min", "30 min", "1 hora", "4 horas"]),
        'risk_level': random.choice(["Bajo", "Medio", "Alto"]),
        'timestamp': datetime.now()
    }

# ===== APLICACIÓN PRINCIPAL =====
def main():
    # ===== HEADER =====
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>🤖 Forex Bot con IA</h1>
            <p style='font-size: 1.2rem; color: #666;'>Predicciones Inteligentes de Divisas</p>
            <p><strong>🟢 Sistema Online</strong> | <strong>⏰ {}</strong></p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.header("⚙️ Panel de Control")
        
        # Estado del sistema
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Estado del Sistema</h3>
            <h2>🟢 OPERATIVO</h2>
            <p>Todos los sistemas funcionando</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Selección de configuración
        st.subheader("🎯 Configuración de Trading")
        
        currency_pairs = load_currency_pairs()
        selected_category = st.selectbox("Categoría:", list(currency_pairs.keys()))
        selected_pairs = st.multiselect(
            "Pares de Divisas:",
            currency_pairs[selected_category],
            default=currency_pairs[selected_category][:2]
        )
        
        time_frame = st.selectbox(
            "Marco Temporal:",
            ["1 minuto", "5 minutos", "15 minutos", "1 hora", "4 horas", "1 día"],
            index=2
        )
        
        risk_level = st.select_slider(
            "Nivel de Riesgo:",
            options=["Conservador", "Moderado", "Agresivo"],
            value="Moderado"
        )
        
        st.markdown("---")
        
        # Botones de acción
        if st.button("🚀 Generar Predicciones", type="primary", use_container_width=True):
            st.session_state.generate_predictions = True
            
        if st.button("📊 Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Datos actualizados")
            st.rerun()
        
        if st.button("⚡ Análisis Express", use_container_width=True):
            st.session_state.express_analysis = True
    
    # ===== MÉTRICAS PRINCIPALES =====
    st.subheader("📈 Métricas en Tiempo Real")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Predicciones Activas</h4>
            <h2>{}</h2>
            <p>En proceso</p>
        </div>
        """.format(len(selected_pairs) if selected_pairs else 0), unsafe_allow_html=True)
    
    with col2:
        accuracy = random.uniform(72, 88)
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Precisión Promedio</h4>
            <h2>{:.1f}%</h2>
            <p>Últimas 24h</p>
        </div>
        """.format(accuracy), unsafe_allow_html=True)
    
    with col3:
        profit = random.uniform(-5, 15)
        color = "#4CAF50" if profit > 0 else "#F44336"
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, {} 0%, {} 100%);">
            <h4>💰 P&L Estimado</h4>
            <h2>{:+.1f}%</h2>
            <p>Este mes</p>
        </div>
        """.format(color, color, profit), unsafe_allow_html=True)
    
    with col4:
        signals = random.randint(3, 12)
        st.markdown("""
        <div class="metric-card">
            <h4>📡 Señales Hoy</h4>
            <h2>{}</h2>
            <p>Ejecutadas</p>
        </div>
        """.format(signals), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== PESTAÑAS PRINCIPALES =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🎯 Predicciones", "📈 Gráficos", "📋 Historial", "⚙️ Configuración"])
    
    # TAB 1: DASHBOARD
    with tab1:
        st.subheader("📊 Dashboard Principal")
        
        if selected_pairs:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Gráfico principal
                st.markdown("#### 📈 Precio en Tiempo Real")
                
                # Generar datos para múltiples pares
                fig = go.Figure()
                
                for pair in selected_pairs[:3]:  # Máximo 3 pares para legibilidad
                    data = generate_sample_data(pair, days=7)
                    fig.add_trace(go.Scatter(
                        x=data['timestamp'],
                        y=data['price'],
                        mode='lines',
                        name=pair,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="Evolución de Precios - Última Semana",
                    xaxis_title="Tiempo",
                    yaxis_title="Precio",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Resumen de Mercado")
                
                for pair in selected_pairs[:5]:
                    data = generate_sample_data(pair, days=1)
                    current_price = data['price'].iloc[-1]
                    prev_price = data['price'].iloc[-2]
                    change = ((current_price - prev_price) / prev_price) * 100
                    
                    if "JPY" in pair:
                        price_str = f"{current_price:.2f}"
                    else:
                        price_str = f"{current_price:.4f}"
                    
                    color = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>{color} {pair}</h4>
                        <h3>{price_str}</h3>
                        <p><strong>{change:+.3f}%</strong> (24h)</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Selecciona pares de divisas en el panel lateral para ver el dashboard")
    
    # TAB 2: PREDICCIONES
    with tab2:
        st.subheader("🎯 Predicciones de IA")
        
        # Generar predicciones automáticamente o por solicitud
        if st.session_state.get('generate_predictions', False) or selected_pairs:
            if selected_pairs:
                st.info("🔄 Generando predicciones con IA...")
                progress_bar = st.progress(0)
                
                predictions = []
                for i, pair in enumerate(selected_pairs):
                    time.sleep(0.1)  # Simular procesamiento
                    predictions.append(generate_prediction(pair))
                    progress_bar.progress((i + 1) / len(selected_pairs))
                
                progress_bar.empty()
                st.success("✅ Predicciones generadas correctamente!")
                
                # Mostrar predicciones
                for i, pred in enumerate(predictions):
                    with st.expander(f"📊 {pred['pair']} - {pred['direction']}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            confidence_color = "#4CAF50" if pred['confidence'] > 0.8 else "#FF9800" if pred['confidence'] > 0.7 else "#F44336"
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: {confidence_color}; color: white; border-radius: 8px;">
                                <h4>Confianza</h4>
                                <h2>{pred['confidence']:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Precio Actual", f"{pred['current_price']}")
                        
                        with col3:
                            change = pred['target_price'] - pred['current_price']
                            st.metric("Precio Objetivo", f"{pred['target_price']}", f"{change:+.4f}")
                        
                        with col4:
                            st.metric("Marco Temporal", pred['time_frame'])
                        
                        # Información adicional
                        st.markdown(f"""
                        **🎯 Recomendación:** {pred['direction']}  
                        **⚠️ Nivel de Riesgo:** {pred['risk_level']}  
                        **⏰ Generado:** {pred['timestamp'].strftime('%H:%M:%S')}
                        """)
                        
                        if st.button(f"📋 Ver Análisis Detallado", key=f"analysis_{i}"):
                            st.markdown(f"""
                            #### 🔍 Análisis Técnico para {pred['pair']}
                            
                            **Indicadores Técnicos:**
                            - RSI: {random.randint(30, 70)}
                            - MACD: {"Alcista" if random.random() > 0.5 else "Bajista"}
                            - Bollinger Bands: {"Sobrecompra" if random.random() > 0.7 else "Normal"}
                            
                            **Análisis Fundamental:**
                            - Volatilidad: {random.choice(["Baja", "Media", "Alta"])}
                            - Tendencia: {random.choice(["Alcista", "Bajista", "Lateral"])}
                            - Soporte/Resistencia: {pred['current_price'] * 0.995:.4f} / {pred['current_price'] * 1.005:.4f}
                            """)
                
                st.session_state.generate_predictions = False
            else:
                st.warning("👈 Selecciona al menos un par de divisas para generar predicciones")
        else:
            st.info("🚀 Haz clic en 'Generar Predicciones' para comenzar el análisis")
    
    # TAB 3: GRÁFICOS
    with tab3:
        st.subheader("📈 Análisis Técnico Avanzado")
        
        if selected_pairs:
            selected_pair_chart = st.selectbox("Selecciona par para análisis:", selected_pairs)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Gráfico de velas japonesas simulado
                data = generate_sample_data(selected_pair_chart, days=30)
                
                # Crear OHLC simulado
                ohlc_data = []
                for i in range(0, len(data), 4):
                    if i + 4 <= len(data):
                        segment = data.iloc[i:i+4]
                        ohlc_data.append({
                            'timestamp': segment['timestamp'].iloc[0],
                            'open': segment['price'].iloc[0],
                            'high': segment['price'].max(),
                            'low': segment['price'].min(),
                            'close': segment['price'].iloc[-1],
                            'volume': segment['volume'].sum()
                        })
                
                ohlc_df = pd.DataFrame(ohlc_data)
                
                fig = go.Figure(data=go.Candlestick(
                    x=ohlc_df['timestamp'],
                    open=ohlc_df['open'],
                    high=ohlc_df['high'],
                    low=ohlc_df['low'],
                    close=ohlc_df['close'],
                    name=selected_pair_chart
                ))
                
                fig.update_layout(
                    title=f"Gráfico de Velas - {selected_pair_chart}",
                    xaxis_title="Tiempo",
                    yaxis_title="Precio",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 Indicadores Técnicos")
                
                # Indicadores simulados
                rsi = random.randint(20, 80)
                rsi_color = "#F44336" if rsi > 70 else "#4CAF50" if rsi < 30 else "#FF9800"
                
                st.markdown(f"""
                <div style="padding: 1rem; background: {rsi_color}; color: white; border-radius: 8px; margin: 0.5rem 0;">
                    <h4>RSI (14)</h4>
                    <h2>{rsi}</h2>
                    <p>{"Sobrecompra" if rsi > 70 else "Sobreventa" if rsi < 30 else "Normal"}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("MACD", "0.0023", "+0.0001")
                st.metric("Estocástico", f"{random.randint(20, 80)}%")
                st.metric("Williams %R", f"{random.randint(-80, -20)}")
                
                # Niveles de soporte y resistencia
                current_price = data['price'].iloc[-1]
                st.markdown(f"""
                **📊 Niveles Clave:**
                - Resistencia: {current_price * 1.01:.4f}
                - Soporte: {current_price * 0.99:.4f}
                - Pivot: {current_price:.4f}
                """)
        else:
            st.info("👈 Selecciona pares de divisas para ver gráficos avanzados")
    
    # TAB 4: HISTORIAL
    with tab4:
        st.subheader("📋 Historial de Operaciones")
        
        # Generar historial simulado
        history_data = []
        for i in range(20):
            pair = random.choice(["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
            direction = random.choice(["COMPRA", "VENTA"])
            profit = random.uniform(-50, 150)
            
            history_data.append({
                'Fecha': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                'Par': pair,
                'Dirección': direction,
                'Precio Entrada': round(random.uniform(1.0, 1.3), 4),
                'Precio Salida': round(random.uniform(1.0, 1.3), 4),
                'P&L': f"${profit:.2f}",
                'Estado': random.choice(['✅ Ganancia', '❌ Pérdida', '⏳ Pendiente'])
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)
        
        # Estadísticas del historial
        col1, col2, col3 = st.columns(3)
        
        with col1:
            win_rate = random.uniform(60, 85)
            st.metric("Tasa de Éxito", f"{win_rate:.1f}%")
        
        with col2:
            total_trades = len(history_data)
            st.metric("Total Operaciones", total_trades)
        
        with col3:
            avg_profit = random.uniform(15, 45)
            st.metric("Ganancia Promedio", f"${avg_profit:.2f}")
    
    # TAB 5: CONFIGURACIÓN
    with tab5:
        st.subheader("⚙️ Configuración del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔑 Configuración de API")
            
            with st.form("api_config"):
                api_key = st.text_input("Alpaca API Key", type="password", help="Tu clave API de Alpaca")
                api_secret = st.text_input("Alpaca Secret Key", type="password", help="Tu clave secreta de Alpaca")
                paper_trading = st.checkbox("Modo Paper Trading", value=True, help="Activar modo de prueba")
                
                if st.form_submit_button("💾 Guardar Configuración", type="primary"):
                    if api_key and api_secret:
                        st.success("✅ Configuración guardada correctamente")
                        st.balloons()
                    else:
                        st.error("❌ Por favor completa todos los campos requeridos")
        
        with col2:
            st.markdown("#### 🎯 Parámetros del Modelo")
            
            confidence_threshold = st.slider("Umbral de Confianza Mínimo", 0.5, 0.95, 0.75, 0.05)
            max_risk = st.slider("Riesgo Máximo por Operación (%)", 1, 10, 2)
            lookback_period = st.number_input("Período de Análisis (días)", 7, 365, 30)
            update_frequency = st.selectbox(
                "Frecuencia de Actualización",
                ["1 minuto", "5 minutos", "15 minutos", "1 hora"]
            )
            
            auto_trading = st.toggle("Trading Automático", value=False)
            
            if auto_trading:
                st.warning("⚠️ El trading automático está activado. Las operaciones se ejecutarán automáticamente.")
            
            st.markdown("#### 📧 Notificaciones")
            
            email_notifications = st.checkbox("Notificaciones por Email", value=True)
            telegram_notifications = st.checkbox("Notificaciones por Telegram", value=False)
            
            if telegram_notifications:
                telegram_token = st.text_input("Token del Bot de Telegram")
                telegram_chat_id = st.text_input("Chat ID de Telegram")
    
    # ===== FOOTER =====
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 Forex Bot v2.0**  
        Desarrollado con ❤️ y IA
        """)
    
    with col2:
        st.markdown("""
        **📊 Estado:** 🟢 Online  
        **⏰ Uptime:** 99.9%
        """)
    
    with col3:
        st.markdown("""
        **📞 Soporte:** [GitHub](https://github.com)  
        **📧 Contacto:** support@forexbot.com
        """)

# ===== EJECUTAR APLICACIÓN =====
if __name__ == "__main__":
    main()
