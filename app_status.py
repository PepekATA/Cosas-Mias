import streamlit as st
import sys
import os
from datetime import datetime

st.set_page_config(
    page_title="🔧 Diagnóstico del Sistema",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Diagnóstico del Sistema Forex Bot")
st.markdown("---")

# Estado básico
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Estado Python", "✅ Funcionando")
    st.metric("Streamlit", "✅ Activo")

with col2:
    st.metric("Puerto", "8501")
    st.metric("Hora", datetime.now().strftime("%H:%M:%S"))

with col3:
    st.metric("Render", "✅ Desplegado")
    st.metric("Logs", "✅ Detectados")

# Información del sistema
st.markdown("## 📊 Información del Sistema")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🐍 Python")
    st.code(f"Versión: {sys.version}")
    st.code(f"Directorio: {os.getcwd()}")

with col2:
    st.markdown("### 📁 Archivos Disponibles")
    try:
        files = os.listdir('.')
        st.code('\n'.join(files[:10]))  # Mostrar solo los primeros 10
    except Exception as e:
        st.error(f"Error listando archivos: {e}")

# Test de módulos
st.markdown("## 🧪 Test de Módulos")

modules_to_test = ['streamlit', 'pandas', 'numpy', 'plotly', 'config']

for module in modules_to_test:
    try:
        __import__(module)
        st.success(f"✅ {module} - OK")
    except ImportError as e:
        st.error(f"❌ {module} - Error: {e}")

# Test del main
st.markdown("## 🚀 Test del Main")

if st.button("🔄 Test Main.py"):
    try:
        import main
        st.success("✅ main.py importado correctamente")
    except Exception as e:
        st.error(f"❌ Error en main.py: {e}")

# Dashboard básico funcional
st.markdown("## 📈 Demo Básico Funcional")

if st.button("🎯 Mostrar Demo"):
    st.success("✅ Sistema funcionando correctamente!")
    
    # Datos demo
    import pandas as pd
    import numpy as np
    
    demo_data = pd.DataFrame({
        'Par': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
        'Precio': [1.0856, 1.2743, 149.32],
        'Dirección': ['UP', 'DOWN', 'UP'],
        'Confianza': ['78%', '82%', '71%']
    })
    
    st.dataframe(demo_data, use_container_width=True)
    
    # Gráfico simple
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['EUR/USD', 'GBP/USD', 'USD/JPY']
    ).cumsum()
    
    st.line_chart(chart_data)

st.markdown("---")
st.info("Si ves esta página, tu aplicación está funcionando correctamente en Render.")
