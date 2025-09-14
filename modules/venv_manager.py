# modules/venv_manager.py
import os
import sys

def check_venv():
    """Verifica si el entorno virtual está activo"""
    if sys.prefix == sys.base_prefix:
        print("⚠️  No estás en un entorno virtual. Activa tu venv antes de ejecutar el proyecto.")
        if os.name == 'nt':
            print(r"venv\Scripts\activate.bat  (CMD) o venv\Scripts\Activate.ps1  (PowerShell)")
        else:
            print("source venv/bin/activate  (Linux/macOS)")
        sys.exit(1)
    else:
        print("✅ Entorno virtual activo:", sys.prefix)
