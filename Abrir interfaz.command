#!/bin/bash
# Doble clic para abrir la interfaz de prueba en el navegador.
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
  echo "Preparando el entorno (solo la primera vez)..."
  python3 -m venv venv && venv/bin/pip install -q -r requirements.txt || exit 1
fi
exec venv/bin/python app.py
