#!/usr/bin/env python3
"""
Interfaz web local para probar paloma.py desde el navegador.

  python app.py            # abre http://127.0.0.1:8765

Usa las mismas funciones que el modo batch (segment, normalize_matrix, tps_warp)
y enseña cada paso: máscara, paloma centrada y resultado final.
Los puntos marcados se pueden guardar como <foto>.json, igual que "paloma.py pick".
"""
import base64
import json
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import paloma

ROOT = Path(__file__).resolve().parent
EJEMPLOS = ROOT / "ejemplos"
HTML = ROOT / "interfaz.html"
PORT = 8765
MAX_SIDE = 1000      # las fotos más grandes se reducen antes de GrabCut (va mucho más rápido)


def encode_jpg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def on_white(rgba):
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_points(img, pts, color, labels=True):
    out = img.copy()
    for name, (x, y) in zip(paloma.REF_POINTS, pts):
        cv2.circle(out, (int(x), int(y)), 10, color, -1, cv2.LINE_AA)
        if labels:
            cv2.putText(out, name, (int(x) + 14, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


def procesar(img, rect=None, points=None):
    """Lo mismo que paloma.process_one, pero devolviendo cada paso como imagen."""
    t0 = time.time()
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        rect = [round(v * s) for v in rect] if rect else None
        points = [[x * s, y * s] for x, y in points] if points else None

    alpha = paloma.segment(img, tuple(rect) if rect else None)

    # Paso 1: la máscara sobre la foto (fondo oscurecido y contorno verde)
    mask_view = (img.astype(np.float32) * 0.25).astype(np.uint8)
    fg = alpha > 127
    mask_view[fg] = img[fg]
    contours, _ = cv2.findContours((fg * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask_view, contours, -1, (0, 255, 0), 3, cv2.LINE_AA)
    if rect:
        x, y, rw, rh = rect
        cv2.rectangle(mask_view, (x, y), (x + rw, y + rh), (26, 159, 255), 3)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    M = paloma.normalize_matrix(alpha)
    C = paloma.CANVAS
    canvas = cv2.warpAffine(rgba, M, (C, C), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    dst = np.array([[x * C, y * C] for x, y in paloma.REF_POINTS.values()])

    # Paso 2: centrada, con los puntos marcados (rojo) y la referencia (azul)
    centrada = on_white(canvas)
    guia = draw_points(centrada, dst, (220, 120, 0))
    final = centrada
    if points:
        if len(points) != len(paloma.REF_POINTS):
            raise ValueError(f"Hacen falta {len(paloma.REF_POINTS)} puntos")
        src = paloma.transform_points(M, points)
        guia = draw_points(guia, src, (0, 0, 230))
        for a, b in zip(src, dst):
            cv2.arrowedLine(guia, tuple(int(v) for v in a), tuple(int(v) for v in b),
                            (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.05)
        final = on_white(paloma.tps_warp(canvas, src, dst))

    return {
        "mascara": encode_jpg(mask_view),
        "centrada": encode_jpg(guia),
        "final": encode_jpg(final),
        "final_ref": encode_jpg(draw_points(final, dst, (220, 120, 0), labels=False)),
        "deformada": bool(points),
        "segundos": round(time.time() - t0, 2),
        "tamano": [img.shape[1], img.shape[0]],
    }


def decode_data_url(url):
    raw = base64.b64decode(url.split(",", 1)[-1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se puede leer la imagen")
    return img


def ejemplo_path(name):
    p = (EJEMPLOS / name).resolve()
    if p.parent != EJEMPLOS or p.suffix.lower() not in paloma.IMG_EXTS or not p.exists():
        raise FileNotFoundError(name)
    return p


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("  " + fmt % args + "\n")

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/config":
            return self.send(200, {"puntos": paloma.REF_POINTS, "canvas": paloma.CANVAS})
        if path == "/api/ejemplos":
            files = sorted(p for p in EJEMPLOS.iterdir() if p.suffix.lower() in paloma.IMG_EXTS)
            out = []
            for p in files:
                j = p.with_suffix(".json")
                pts = json.loads(j.read_text())["puntos"] if j.exists() else None
                out.append({"nombre": p.name, "puntos": pts})
            return self.send(200, out)
        if path.startswith("/ejemplos/"):
            try:
                p = ejemplo_path(path[len("/ejemplos/"):])
            except FileNotFoundError:
                return self.send(404, {"error": "No existe"})
            return self.send(200, p.read_bytes(), "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png")
        self.send(404, {"error": "No existe"})

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/api/procesar":
                img = decode_data_url(data["imagen"])
                return self.send(200, procesar(img, data.get("rect"), data.get("puntos")))
            if self.path == "/api/guardar-puntos":
                p = ejemplo_path(data["nombre"])
                pts = [[int(round(x)), int(round(y))] for x, y in data["puntos"]]
                p.with_suffix(".json").write_text(json.dumps({"puntos": pts}, indent=2))
                return self.send(200, {"guardado": p.with_suffix(".json").name})
            self.send(404, {"error": "No existe"})
        except Exception as e:
            self.send(400, {"error": str(e)})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"PigeonPhotoMaketation en {url}  (Ctrl+C para parar)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
