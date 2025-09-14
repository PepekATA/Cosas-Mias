#!/bin/bash
echo "🚀 Preparando deployment del Bot de Predicción de Divisas"

# Verificar archivos necesarios
echo "📋 Verificando archivos necesarios..."

required_files=(
    "Dockerfile"
    "requirements.txt" 
    "main.py"
    "config.py"
    "modules/__init__.py"
    "render.yaml"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Archivos faltantes:"
    printf '%s\n' "${missing_files[@]}"
    exit 1
fi

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p data/historical data/models logs

# Crear archivos .gitkeep
touch data/historical/.gitkeep
touch data/models/.gitkeep
touch logs/.gitkeep

# Verificar variables de entorno
echo "🔐 Verificando variables de entorno..."
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "⚠️  Advertencia: Variables de entorno ALPACA_API_KEY y ALPACA_SECRET_KEY no están configuradas"
    echo "   Asegúrate de configurarlas en tu plataforma de deployment"
fi

# Commit y push si es repositorio git
if [ -d ".git" ]; then
    echo "📤 Subiendo cambios a repositorio..."
    git add .
    git commit -m "Add deployment files for forex prediction bot"
    git push
fi

echo "✅ Preparación completada"
echo ""
echo "🌐 Para deployar en diferentes plataformas:"
echo "   Render: Conecta tu repo en https://render.com"
echo "   Railway: railway login && railway up"
echo "   Heroku: git push heroku main"
echo "   Docker: docker build -t forex-bot . && docker run -p 8501:8501 forex-bot"
