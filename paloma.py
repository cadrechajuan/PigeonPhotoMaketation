#!/usr/bin/env python3
"""
Pipeline para estandarizar fotos de palomas:
  1. Quita el fondo (GrabCut, sin IA generativa)
  2. Escala y centra la paloma en un lienzo cuadrado
  3. Deforma (opcional) puntos clave hacia la posición de referencia
  4. Guarda sobre fondo blanco uniforme

Uso:
  python paloma.py pick  foto.jpg               # marcar puntos clave (genera foto.json)
  python paloma.py batch carpeta_in carpeta_out # procesar todas las fotos
  python paloma.py batch carpeta_in carpeta_out --rect 100,80,1500,1800
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
CANVAS = 1200        # lado del resultado en píxeles
MARGIN = 0.06        # margen alrededor de la paloma (fracción del lienzo)
GRID_STEP = 4        # resolución de la malla de deformación (más alto = más rápido)

# Puntos clave en orden. Coordenadas normalizadas (0..1) de la imagen de referencia.
# AJÚSTALAS mirando tu foto de referencia: ancho/alto relativos de cada punto.
REF_POINTS = {
    "pico":      [0.93, 0.12],
    "cola":      [0.05, 0.83],
    "pata_der":  [0.87, 0.95],
    "pata_izq":  [0.61, 0.94],
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 1. SEGMENTACIÓN
# ---------------------------------------------------------------------------
def segment(img_bgr, rect=None):
    """Devuelve una máscara alfa (0/255) con la paloma."""
    h, w = img_bgr.shape[:2]
    if rect is None:
        m = int(0.03 * min(h, w))
        rect = (m, m, w - 2 * m, h - 2 * m)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Conservar solo la región más grande (la paloma)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg = np.where(labels == big, 255, 0).astype(np.uint8)

    # Rellenar huecos y suavizar bordes
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.GaussianBlur(fg, (3, 3), 0)
    return fg


# ---------------------------------------------------------------------------
# 2. NORMALIZACIÓN (escala + centrado)
# ---------------------------------------------------------------------------
def normalize_matrix(alpha):
    ys, xs = np.nonzero(alpha > 127)
    if xs.size == 0:
        raise ValueError("No se detectó la paloma en la imagen")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    size = max(x1 - x0, y1 - y0)
    scale = (CANVAS * (1 - 2 * MARGIN)) / size
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    M = np.array([[scale, 0, CANVAS / 2 - scale * cx],
                  [0, scale, CANVAS / 2 - scale * cy]], np.float64)
    return M


def transform_points(M, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.transform(pts, M).reshape(-1, 2)


# ---------------------------------------------------------------------------
# 3. DEFORMACIÓN LIBRE (thin plate spline sobre puntos clave)
# ---------------------------------------------------------------------------
def tps_warp(img_rgba, src, dst):
    """
    src: puntos actuales en el lienzo normalizado (píxeles)
    dst: posiciones destino (píxeles del lienzo final)
    Las esquinas se fijan para que el fondo no se deforme.
    """
    c = CANVAS - 1
    corners = np.array([[0, 0], [c, 0], [0, c], [c, c],
                        [c / 2, 0], [c / 2, c], [0, c / 2], [c, c / 2]], np.float64)
    dst_all = np.vstack([dst, corners])
    src_all = np.vstack([src, corners])

    # Mapeo inverso: para cada píxel destino, de dónde sacar el color
    rbf_x = RBFInterpolator(dst_all, src_all[:, 0], kernel="thin_plate_spline")
    rbf_y = RBFInterpolator(dst_all, src_all[:, 1], kernel="thin_plate_spline")

    gs = CANVAS // GRID_STEP
    gx, gy = np.meshgrid(np.linspace(0, c, gs), np.linspace(0, c, gs))
    q = np.column_stack([gx.ravel(), gy.ravel()])
    mx = rbf_x(q).reshape(gs, gs)
    my = rbf_y(q).reshape(gs, gs)

    map_x = cv2.resize(mx.astype(np.float32), (CANVAS, CANVAS), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(my.astype(np.float32), (CANVAS, CANVAS), interpolation=cv2.INTER_LINEAR)

    return cv2.remap(img_rgba, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


# ---------------------------------------------------------------------------
# 4. PROCESADO DE UNA FOTO
# ---------------------------------------------------------------------------
def process_one(path, out_dir, rect=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"No se puede leer {path}")

    alpha = segment(img, rect)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    M = normalize_matrix(alpha)
    canvas = cv2.warpAffine(rgba, M, (CANVAS, CANVAS),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Deformación opcional si existe el JSON con los puntos marcados
    lm_file = path.with_suffix(".json")
    if lm_file.exists():
        data = json.loads(lm_file.read_text())
        pts = data["puntos"]
        if len(pts) != len(REF_POINTS):
            raise ValueError(f"{lm_file.name}: se esperaban {len(REF_POINTS)} puntos")
        src = transform_points(M, pts)
        dst = np.array([[x * CANVAS, y * CANVAS] for x, y in REF_POINTS.values()])
        canvas = tps_warp(canvas, src, dst)

    # Componer sobre fondo blanco
    a = canvas[:, :, 3:4].astype(np.float32) / 255.0
    rgb = canvas[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)
    out = np.clip(rgb, 0, 255).astype(np.uint8)

    out_path = out_dir / (path.stem + ".jpg")
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_path


# ---------------------------------------------------------------------------
# MODO "PICK": marcar puntos clave con el ratón
# ---------------------------------------------------------------------------
def pick(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"No se puede leer {path}")
    names = list(REF_POINTS)
    clicks = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(names):
            clicks.append([x, y])

    win = "pick: ENTER guarda, BACKSPACE borra, ESC cancela"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        view = img.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(view, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(view, names[i], (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if len(clicks) < len(names):
            cv2.putText(view, f"Haz clic en: {names[len(clicks)]}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow(win, view)
        k = cv2.waitKey(20)
        if k == 27:
            break
        if k in (10, 13) and len(clicks) == len(names):
            path.with_suffix(".json").write_text(json.dumps({"puntos": clicks}, indent=2))
            print(f"Guardado {path.with_suffix('.json').name}")
            break
        if k in (8, 127) and clicks:
            clicks.pop()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# MODO "BATCH"
# ---------------------------------------------------------------------------
def batch(in_dir, out_dir, rect=None):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    ok, fail = 0, 0
    for f in files:
        try:
            out = process_one(f, out_dir, rect)
            print(f"OK   {f.name} -> {out.name}")
            ok += 1
        except Exception as e:
            print(f"ERR  {f.name}: {e}")
            fail += 1
    print(f"\nTerminado: {ok} correctas, {fail} con error.")


def main():
    ap = argparse.ArgumentParser(description="Estandarizar fotos de palomas")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pick", help="marcar puntos clave de una foto")
    p1.add_argument("foto")

    p2 = sub.add_parser("batch", help="procesar una carpeta")
    p2.add_argument("entrada")
    p2.add_argument("salida")
    p2.add_argument("--rect", help="x,y,ancho,alto del recuadro de la paloma (opcional)")

    a = ap.parse_args()
    if a.cmd == "pick":
        pick(Path(a.foto))
    else:
        rect = tuple(int(v) for v in a.rect.split(",")) if a.rect else None
        batch(a.entrada, a.salida, rect)


if __name__ == "__main__":
    main()
