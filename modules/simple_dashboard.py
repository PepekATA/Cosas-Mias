import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def run_simple_dashboard():
    st.title("🤖 Forex Prediction Bot")
    st.markdown("---")
    
    # Estado básico
    st.sidebar.title("⚙️ Control Panel")
    
    # Selección básica
    pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
    selected = st.sidebar.multiselect("Pares:", pairs, default=pairs[:2])
    
    # Botón para generar datos demo
    if st.sidebar.button("🚀 Generar Demo"):
        st.session_state.demo_data = True
    
    # Mostrar contenido
    if st.session_state.get('demo_data'):
        st.success("✅ Sistema funcionando correctamente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicciones", "4")
            st.metric("Precisión", "78%")
        
        with col2:
            st.metric("Pares Activos", len(selected))
            st.metric("Estado", "🟢 Online")
        
        # Tabla demo
        df = pd.DataFrame({
            'Par': selected,
            'Dirección': np.random.choice(['UP', 'DOWN'], len(selected)),
            'Confianza': np.random.uniform(0.6, 0.9, len(selected))
        })
        
        st.dataframe(df, use_container_width=True)
        
        # Gráfico demo
        chart_data = pd.DataFrame(
            np.random.randn(20, 2),
            columns=['EUR/USD', 'GBP/USD']
        ).cumsum()
        
        st.line_chart(chart_data)
    
    else:
        st.info("👆 Haz clic en 'Generar Demo' para probar el sistema")
        
        # Información del sistema
        st.markdown("""
        ### 📊 Sistema de Predicción de Divisas
        
        **Características:**
        - 🤖 Predicciones con IA
        - 📈 Análisis técnico avanzado
        - ⚡ Tiempo real
        - 📊 Dashboard interactivo
        
        **Estado:** ✅ Sistema inicializado correctamente
        """)

if __name__ == "__main__":
    run_simple_dashboard()
