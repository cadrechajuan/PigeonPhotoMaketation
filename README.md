# PigeonPhotoMaketation

Pipeline en Python para estandarizar fotos de palomas sobre fondo blanco.
Usa solo procesado de imagen clásico (OpenCV y SciPy), sin IA generativa ni filtros.

## Qué hace

1. Quita el fondo con GrabCut.
2. Escala y centra la paloma en un lienzo cuadrado.
3. Opcionalmente deforma zonas concretas (pico, cola, patas) hacia una posición de referencia.
4. Guarda el resultado sobre fondo blanco.

## Instalación

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# 1. Marca los puntos clave en una foto de ejemplo (genera un .json junto a la foto)
python paloma.py pick fotos/paloma01.jpg

# 2. Procesa toda la carpeta
python paloma.py batch fotos resultados

# Opcional: recuadro de la paloma para ayudar a GrabCut
python paloma.py batch fotos resultados --rect 100,80,1500,1800
```

## Interfaz de prueba

Doble clic en `Abrir interfaz.command` (o `python app.py`) abre http://127.0.0.1:8765 en el navegador.
Elige una foto de `ejemplos/` o sube una tuya, marca opcionalmente un recuadro y los 4 puntos clave,
y pulsa «Procesar» para ver la máscara, la paloma centrada y el resultado final.
«Guardar puntos» escribe el mismo `.json` que `paloma.py pick`, así que luego sirve para `batch`.

## Notas

- Las coordenadas de referencia (`REF_POINTS` en `paloma.py`) son aproximadas y deben ajustarse a tu foto de referencia.
- Para mejores máscaras, fotografía sobre un fondo liso y contrastado.
