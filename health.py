"""
Endpoint de salud para verificar el estado de la aplicación
"""

def health_check():
    """
    Verificación básica de salud de la aplicación
    """
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        
        # Verificar que los módulos principales se pueden importar
        from modules import ForexPredictor, DataStorage
        
        return {
            "status": "healthy",
            "timestamp": pd.Timestamp.now().isoformat(),
            "modules": "loaded"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

if __name__ == "__main__":
    result = health_check()
    print(result)
