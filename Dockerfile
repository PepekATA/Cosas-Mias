# Usar Python 3.11 como base
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/historical data/models logs

# Exponer puerto
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--", "--mode", "dashboard"]
