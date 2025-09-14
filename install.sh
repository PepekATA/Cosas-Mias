#!/bin/bash
echo "🚀 Instalando Bot de Predicción de Divisas"

# Crear entorno virtual
python -m venv forex_bot_env

# Activar entorno virtual
source forex_bot_env/bin/activate  # Linux/Mac
# forex_bot_env\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Crear directorios necesarios
mkdir -p data/historical
mkdir -p data/models
mkdir -p logs

echo "✅ Instalación completada"
echo "📝 Recuerda configurar tu archivo .env con las claves de API"
echo "🏃 Para ejecutar: python main.py --mode predict"
echo "🖥️  Para dashboard: python main.py --mode dashboard"
