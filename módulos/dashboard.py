import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from .predictor import ForexPredictor
from .storage import DataStorage
from config import CURRENCY_PAIRS, PREDICTION_INTERVALS

class Dashboard:
    def __init__(self):
        self.predictor = ForexPredictor()
        self.storage = DataStorage()
        
    def run(self):
        """
        Ejecuta la aplicación Streamlit
        """
        st.set_page_config(
            page_title="Forex Prediction Bot",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 Bot de Predicción de Divisas")
        st.markdown("---")
        
        # Sidebar para controles
        self.create_sidebar()
        
        # Contenido principal
        self.create_main_content()
    
    def create_sidebar(self):
        """
        Crea la barra lateral con controles
        """
        st.sidebar.header("⚙️ Configuración")
        
        # Selección de pares de divisas
        st.sidebar.subheader("Pares de Divisas")
        selected_pairs = st.sidebar.multiselect(
            "Selecciona pares para analizar:",
            CURRENCY_PAIRS,
            default=CURRENCY_PAIRS[:4]
        )
        
        # Intervalo de predicción
        st.sidebar.subheader("Intervalo de Predicción")
        selected_interval = st.sidebar.selectbox(
            "Selecciona intervalo:",
            list(PREDICTION_INTERVALS.keys()),
            index=2  # 5m por defecto
        )
        
        # Botones de acción
        st.sidebar.subheader("Acciones")
        
        if st.sidebar.button("🔄 Actualizar Predicciones", type="primary"):
            self.update_predictions(selected_pairs, selected_interval)
        
        if st.sidebar.button("📊 Entrenar Modelos"):
            self.retrain_models(selected_pairs)
        
        if st.sidebar.button("🧹 Limpiar Cache"):
            st.cache_data.clear()
            st.success("Cache limpiado")
        
        # Guardar selecciones en session state
        st.session_state['selected_pairs'] = selected_pairs
        st.session_state['selected_interval'] = selected_interval
    
    def create_main_content(self):
        """
        Crea el contenido principal del dashboard
        """
        # Obtener configuración
        selected_pairs = st.session_state.get('selected_pairs', CURRENCY_PAIRS[:4])
        selected_interval = st.session_state.get('selected_interval', '5m')
        
        # Tabs principales
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Predicciones Actuales", 
            "📊 Análisis Técnico", 
            "🎯 Rendimiento", 
            "📋 Historial"
        ])
        
        with tab1:
            self.show_current_predictions(selected_pairs, selected_interval)
        
        with tab2:
            self.show_technical_analysis(selected_pairs)
        
        with tab3:
            self.show_performance_metrics()
        
        with tab4:
            self.show_prediction_history()
    
    def show_current_predictions(self, pairs, interval):
        """
        Muestra predicciones actuales
        """
        st.header(f"Predicciones para intervalo: {interval}")
        
        # Obtener predicciones
        if st.button("🚀 Generar Nuevas Predicciones"):
            with st.spinner("Generando predicciones..."):
                predictions = self.get_predictions(pairs, interval)
            
            # Mostrar resultados
            if predictions:
                self.display_predictions(predictions)
            else:
                st.error("No se pudieron generar predicciones")
        
        # Mostrar última predicción guardada si existe
        st.subheader("Últimas Predicciones")
        self.show_latest_predictions(pairs)
    
    def display_predictions(self, predictions):
        """
        Muestra las predicciones en formato visual
        """
        cols = st.columns(2)
        
        for i, prediction in enumerate(predictions):
            col = cols[i % 2]
            
            with col:
                # Card de predicción
                self.create_prediction_card(prediction)
    
    def create_prediction_card(self, prediction):
        """
        Crea una tarjeta visual para mostrar una predicción
        """
        symbol = prediction['symbol']
        direction = prediction['direction']
        confidence = prediction['confidence']
        current_price = prediction['current_price']
        price_target = prediction['price_target']
        duration = prediction['duration_minutes']
        
        # Color según dirección
        if direction == 'UP':
            color = "green"
            arrow = "⬆️"
            bg_color = "#d4edda"
        elif direction == 'DOWN':
            color = "red"
            arrow = "⬇️"
            bg_color = "#f8d7da"
        else:
            color = "gray"
            arrow = "➡️"
            bg_color = "#e2e3e5"
        
        # HTML para la tarjeta
        card_html = f"""
        <div style="
            padding: 20px;
            border-radius: 10px;
            background-color: {bg_color};
            border-left: 5px solid {color};
            margin: 10px 0;
        ">
            <h3 style="color: {color}; margin: 0;">{arrow} {symbol}</h3>
            <p style="font-size: 18px; margin: 5px 0;"><strong>Dirección: {direction}</strong></p>
            <p style="margin: 5px 0;">Confianza: <strong>{confidence:.1%}</strong></p>
            <p style="margin: 5px 0;">Precio Actual: <strong>{current_price:.5f}</strong></p>
            <p style="margin: 5px 0;">Precio Objetivo: <strong>{price_target:.5f}</strong></p>
            <p style="margin: 5px 0;">Duración Estimada: <strong>{duration} minutos</strong></p>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Gráfica de precio con objetivo
        self.create_price_chart(symbol, current_price, price_target, direction)
    
    def create_price_chart(self, symbol, current_price, target_price, direction):
        """
        Crea gráfica de precio con objetivo
        """
        try:
            # Obtener datos históricos recientes
            historical_data = self.storage.load_historical_data(symbol)
            
            if not historical_data.empty:
                # Últimos 100 puntos
                recent_data = historical_data.tail(100).copy()
                
                # Crear gráfica con candlesticks
                fig = go.Figure()
                
                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['open'],
                    high=recent_data['high'],
                    low=recent_data['low'],
                    close=recent_data['close'],
                    name=symbol
                ))
                
                # Línea de precio actual
                fig.add_hline(
                    y=current_price, 
                    line_dash="dash", 
                    line_color="blue",
                    annotation_text=f"Precio Actual: {current_price:.5f}"
                )
                
                # Línea de precio objetivo
                color = "green" if direction == "UP" else "red"
                fig.add_hline(
                    y=target_price, 
                    line_dash="dot", 
                    line_color=color,
                    annotation_text=f"Objetivo: {target_price:.5f}"
                )
                
                # Configurar layout
                fig.update_layout(
                    title=f"{symbol} - Predicción {direction}",
                    yaxis_title="Precio",
                    xaxis_title="Tiempo",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creando gráfica para {symbol}: {e}")
    
    def show_technical_analysis(self, pairs):
        """
        Muestra análisis técnico detallado
        """
        st.header("Análisis Técnico")
        
        selected_pair = st.selectbox("Selecciona par para análisis:", pairs)
        
        if selected_pair:
            # Obtener datos con indicadores
            historical_data = self.storage.load_historical_data(selected_pair)
            
            if not historical_data.empty:
                # Crear gráfica completa con indicadores
                self.create_technical_chart(selected_pair, historical_data)
                
                # Mostrar métricas de indicadores
                self.show_technical_metrics(historical_data)
            else:
                st.warning(f"No hay datos históricos para {selected_pair}")
    
    def create_technical_chart(self, symbol, df):
        """
        Crea gráfica técnica completa con indicadores
        """
        # Datos recientes
        recent_data = df.tail(200)
        
        # Crear subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{symbol} - Precio y Medias Móviles",
                "RSI",
                "MACD",
                "Bollinger Bands"
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. Precio con medias móviles
        fig.add_trace(go.Candlestick(
            x=recent_data.index,
            open=recent_data['open'],
            high=recent_data['high'],
            low=recent_data['low'],
            close=recent_data['close'],
            name="Precio"
        ), row=1, col=1)
        
        # Medias móviles
        if 'EMA_20' in recent_data.columns:
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['EMA_20'],
                line=dict(color='orange', width=1),
                name="EMA 20"
            ), row=1, col=1)
        
        if 'EMA_50' in recent_data.columns:
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['EMA_50'],
                line=dict(color='red', width=1),
                name="EMA 50"
            ), row=1, col=1)
        
        # 2. RSI
        if 'RSI' in recent_data.columns:
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['RSI'],
                line=dict(color='purple'),
                name="RSI"
            ), row=2, col=1)
            
            # Líneas de sobrecompra/sobreventa
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 3. MACD
        if 'MACD' in recent_data.columns:
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['MACD'],
                line=dict(color='blue'),
                name="MACD"
            ), row=3, col=1)
            
            if 'MACD_signal' in recent_data.columns:
                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['MACD_signal'],
                    line=dict(color='red'),
                    name="MACD Signal"
                ), row=3, col=1)
        
        # 4. Bollinger Bands
        if all(col in recent_data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['BB_upper'],
                line=dict(color='gray', width=1),
                name="BB Upper"
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['BB_middle'],
                line=dict(color='blue', width=1),
                name="BB Middle"
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['BB_lower'],
                line=dict(color='gray', width=1),
                name="BB Lower"
            ), row=4, col=1)
        
        # Configurar layout
        fig.update_layout(
            height=800,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_technical_metrics(self, df):
        """
        Muestra métricas actuales de indicadores técnicos
        """
        st.subheader("Métricas Actuales")
        
        # Crear columnas para métricas
        col1, col2, col3, col4 = st.columns(4)
        
        latest = df.iloc[-1]
        
        with col1:
            if 'RSI' in df.columns:
                rsi_value = latest['RSI']
                rsi_status = "Sobrevendido" if rsi_value < 30 else "Sobrecomprado" if rsi_value > 70 else "Normal"
                rsi_color = "green" if rsi_value < 30 else "red" if rsi_value > 70 else "blue"
                
                st.metric(
                    label="RSI",
                    value=f"{rsi_value:.2f}",
                    delta=rsi_status
                )
        
        with col2:
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd_value = latest['MACD']
                macd_signal = latest['MACD_signal']
                macd_status = "Alcista" if macd_value > macd_signal else "Bajista"
                
                st.metric(
                    label="MACD",
                    value=f"{macd_value:.6f}",
                    delta=macd_status
                )
        
        with col3:
            if 'BB_position' in df.columns:
                bb_pos = latest['BB_position']
                bb_status = "Cerca inf." if bb_pos < 0.2 else "Cerca sup." if bb_pos > 0.8 else "Centro"
                
                st.metric(
                    label="Posición BB",
                    value=f"{bb_pos:.2f}",
                    delta=bb_status
                )
        
        with col4:
            if 'volatility_5' in df.columns:
                volatility = latest['volatility_5'] * 100
                vol_status = "Alta" if volatility > 0.1 else "Baja" if volatility < 0.05 else "Media"
                
                st.metric(
                    label="Volatilidad %",
                    value=f"{volatility:.3f}",
                    delta=vol_status
                )
    
    def show_performance_metrics(self):
        """
        Muestra métricas de rendimiento de los modelos
        """
        st.header("Rendimiento de Modelos")
        
        # Obtener resumen de rendimiento
        performance = self.storage.get_performance_summary()
        
        if performance['total_predictions'] > 0:
            # Métricas generales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Predicciones",
                    value=performance['total_predictions']
                )
            
            with col2:
                st.metric(
                    label="Predicciones Correctas",
                    value=performance['correct_predictions']
                )
            
            with col3:
                st.metric(
                    label="Precisión General",
                    value=f"{performance['overall_accuracy']:.1%}"
                )
            
            # Rendimiento por símbolo
            st.subheader("Rendimiento por Par de Divisas")
            
            if 'by_symbol' in performance:
                symbol_data = []
                for symbol, stats in performance['by_symbol'].items():
                    symbol_data.append({
                        'Símbolo': symbol,
                        'Total': stats['total'],
                        'Correctas': stats['correct'],
                        'Precisión': f"{stats['accuracy']:.1%}",
                        'Error Promedio': f"{stats['avg_price_error']:.4f}"
                    })
                
                df_performance = pd.DataFrame(symbol_data)
                st.dataframe(df_performance, use_container_width=True)
                
                # Gráfica de precisión por símbolo
                fig = px.bar(
                    df_performance,
                    x='Símbolo',
                    y='Precisión',
                    title="Precisión por Par de Divisas"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No hay datos de rendimiento disponibles aún.")
    
    def show_prediction_history(self):
        """
        Muestra historial de predicciones
        """
        st.header("Historial de Predicciones")
        
        # Obtener predicciones pasadas
        past_predictions = self.storage.get_past_predictions(days_back=7)
        
        if past_predictions:
            # Convertir a DataFrame para mejor visualización
            df_predictions = pd.DataFrame(past_predictions)
            
            # Filtros
            col1, col2 = st.columns(2)
            
            with col1:
                selected_symbol = st.selectbox(
                    "Filtrar por símbolo:",
                    ['Todos'] + list(df_predictions['symbol'].unique())
                )
            
            with col2:
                selected_direction = st.selectbox(
                    "Filtrar por dirección:",
                    ['Todas', 'UP', 'DOWN', 'NEUTRAL']
                )
            
            # Aplicar filtros
            filtered_df = df_predictions.copy()
            
            if selected_symbol != 'Todos':
                filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
            
            if selected_direction != 'Todas':
                filtered_df = filtered_df[filtered_df['direction'] == selected_direction]
            
            # Mostrar tabla
            display_columns = [
                'timestamp', 'symbol', 'direction', 'confidence',
                'current_price', 'price_target', 'duration_minutes'
            ]
            
            st.dataframe(
                filtered_df[display_columns].sort_values('timestamp', ascending=False),
                use_container_width=True
            )
            
            # Estadísticas del período
            st.subheader("Estadísticas del Período")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predicciones", len(filtered_df))
            
            with col2:
                up_predictions = len(filtered_df[filtered_df['direction'] == 'UP'])
                st.metric("Predicciones UP", up_predictions)
            
            with col3:
                down_predictions = len(filtered_df[filtered_df['direction'] == 'DOWN'])
                st.metric("Predicciones DOWN", down_predictions)
            
            with col4:
                avg_confidence = filtered_df['confidence'].mean()
                st.metric("Confianza Promedio", f"{avg_confidence:.1%}")
        
        else:
            st.info("No hay historial de predicciones disponible.")
    
    @st.cache_data(ttl=300)  # Cache por 5 minutos
    def get_predictions(_self, pairs, interval):
        """
        Obtiene predicciones (con cache)
        """
        try:
            # Inicializar modelos si no existen
            _self.predictor.initialize_models(pairs)
            
            # Generar predicciones
            predictions = _self.predictor.predict_multiple_pairs(pairs, interval)
            
            return predictions
            
        except Exception as e:
            st.error(f"Error obteniendo predicciones: {e}")
            return []
    
    def show_latest_predictions(self, pairs):
        """
        Muestra las últimas predicciones guardadas
        """
        recent_predictions = self.storage.get_past_predictions(days_back=1)
        
        if recent_predictions:
            # Filtrar por pares seleccionados
            filtered_predictions = [
                p for p in recent_predictions 
                if p['symbol'] in pairs
            ]
            
            if filtered_predictions:
                st.subheader("Predicciones del Día")
                
                # Mostrar las más recientes
                for prediction in sorted(filtered_predictions, key=lambda x: x['timestamp'], reverse=True)[:4]:
                    with st.expander(f"{prediction['symbol']} - {prediction['timestamp'].strftime('%H:%M')}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Dirección:** {prediction['direction']}")
                            st.write(f"**Confianza:** {prediction['confidence']:.1%}")
                        
                        with col2:
                            st.write(f"**Precio Actual:** {prediction['current_price']:.5f}")
                            st.write(f"**Precio Objetivo:** {prediction['price_target']:.5f}")
                        
                        with col3:
                            st.write(f"**Duración:** {prediction['duration_minutes']} min")
                            st.write(f"**Intervalo:** {prediction['interval']}")
    
    def update_predictions(self, pairs, interval):
        """
        Actualiza predicciones y muestra resultados
        """
        with st.spinner("Actualizando predicciones..."):
            try:
                predictions = self.get_predictions(pairs, interval)
                
                if predictions:
                    st.success("Predicciones actualizadas exitosamente!")
                    self.display_predictions(predictions)
                else:
                    st.error("No se pudieron generar predicciones")
                    
            except Exception as e:
                st.error(f"Error actualizando predicciones: {e}")
    
    def retrain_models(self, pairs):
        """
        Reentrena modelos para los pares seleccionados
        """
        with st.spinner("Entrenando modelos..."):
            try:
                results = {}
                
                for pair in pairs:
                    st.write(f"Entrenando modelo para {pair}...")
                    success = self.predictor.train_model(pair)
                    results[pair] = success
                
                # Mostrar resultados
                success_count = sum(results.values())
                total_count = len(results)
                
                if success_count == total_count:
                    st.success(f"Todos los modelos entrenados exitosamente! ({success_count}/{total_count})")
                else:
                    st.warning(f"Entrenamiento completado: {success_count}/{total_count} modelos exitosos")
                
                # Detalles por par
                for pair, success in results.items():
                    status = "✅" if success else "❌"
                    st.write(f"{status} {pair}")
                    
            except Exception as e:
                st.error(f"Error entrenando modelos: {e}")

# Función para ejecutar la aplicación
def run_dashboard():
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    run_dashboard()
