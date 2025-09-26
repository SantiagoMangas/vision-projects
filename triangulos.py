import cv2
import numpy as np

# --- Función auxiliar: distancia de un punto a una línea (p, a, b vectores float) ---
def dist_punto_a_linea(p, a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    p = np.array(p, dtype=np.float32)
    denom = np.linalg.norm(b - a)
    if denom == 0:
        return 0.0
    return abs(np.cross(b - a, p - a)) / denom

# --- Ordenar 4 vértices en orden: top-left, top-right, bottom-right, bottom-left
def ordenar_vertices_por_posicion(vertices):
    vertices = np.array(vertices, dtype=np.float32)
    centro = np.mean(vertices, axis=0)
    angulos = [np.arctan2(v[1] - centro[1], v[0] - centro[0]) for v in vertices]
    # combinar y ordenar por ángulo (de -pi a pi). Queremos clockwise empezando en top-left
    paired = sorted(zip(angulos, vertices), key=lambda x: x[0])
    verts_sorted = [v for _, v in paired]
    # A veces la secuencia empieza en otro vértice: rotamos para que empiece en top-left (min x+y)
    sums = [v[0] + v[1] for v in verts_sorted]
    start_idx = int(np.argmin(sums))
    verts_sorted = verts_sorted[start_idx:] + verts_sorted[:start_idx]
    return np.array(verts_sorted, dtype=np.float32)

# --- Calcular cuarto vértice probando paralelogramos y seleccionando el mejor rectángulo ---
def calcular_cuarto_vertice_mejorado(puntos):
    puntos = [np.array(p, dtype=np.float32) for p in puntos]
    mejor_score = -1
    mejor_rect = None
    mejor_D = None

    # probar permutaciones simples: tomar combinaciones A,B,C y generar D = A + C - B
    from itertools import permutations
    for (A, B, C) in permutations(puntos, 3):
        D = A + C - B
        # descartar fuera de imagen razonable (si tu imagen es más grande, ajustar)
        if np.any(D < -100) or np.any(D > 10000):
            continue
        cand = [A, B, C, D]
        cand_ord = ordenar_vertices_por_posicion(cand)
        score = evaluar_rectangulo(cand_ord)
        if score > mejor_score:
            mejor_score = score
            mejor_rect = cand_ord
            mejor_D = D

    return mejor_D, mejor_rect

# --- Evaluar rectángulo (score mayor = mejor) ---
def evaluar_rectangulo(vertices):
    if vertices is None or len(vertices) != 4:
        return -1
    v = np.array(vertices, dtype=np.float32)
    # 1) ángulos próximos a 90
    score = 0.0
    for i in range(4):
        p0 = v[i]
        p1 = v[(i + 1) % 4]
        p2 = v[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return -1
        cosang = np.dot(v1, v2) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        dev = abs(ang - 90)
        if dev > 40:  # demasiado malo
            return -1
        score += max(0, (40 - dev)) / 40.0

    # 2) lados opuestos similares
    l0 = np.linalg.norm(v[1] - v[0])
    l1 = np.linalg.norm(v[2] - v[1])
    l2 = np.linalg.norm(v[3] - v[2])
    l3 = np.linalg.norm(v[0] - v[3])
    if max(l0, l2) == 0 or max(l1, l3) == 0:
        return -1
    diff_h = abs(l0 - l2) / max(l0, l2)
    diff_v = abs(l1 - l3) / max(l1, l3)
    if diff_h > 0.5 or diff_v > 0.5:
        return -1
    score += (1 - diff_h) + (1 - diff_v)

    # 3) proporción A4 aproximada
    ancho = (l0 + l2) / 2.0
    alto = (l1 + l3) / 2.0
    if min(ancho, alto) == 0:
        return -1
    ratio = max(ancho, alto) / min(ancho, alto)
    ratio_a4 = 297.0 / 210.0
    score += max(0, 1 - abs(ratio - ratio_a4) / ratio_a4)

    return score

# --- Cargar imagen ---
img = cv2.imread("hoja4.jpg")
if img is None:
    raise ValueError("No se pudo cargar la imagen")

# Ventanas
cv2.namedWindow("Deteccion", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Deteccion", 900, 700)
cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thresh", 600, 700)

cv2.createTrackbar("Umbral", "Thresh", 23, 255, lambda x: None)  # vos decís que 23 anda bien

while True:
    umbral = cv2.getTrackbarPos("Umbral", "Thresh")

    # Preprocesamiento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu
    otsu_val, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # offset sobre Otsu (usamos el trackbar centrado en 0)
    offset = umbral - 128
    thresh_val = int(max(0, min(255, otsu_val + offset)))
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # limpieza morfológica para unir pequeños huecos
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    puntos_referencia = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:  # ajustar si tus triángulos son más chicos/grandes
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # permitir triángulos "aproximados" (len==3) o 4 vértices muy simplificados
        if len(approx ) == 3:
            pts = approx.reshape(-1, 2)
            # elegir vértice opuesto al lado más largo (tu heurística)
            lados = [
                (np.linalg.norm(pts[0] - pts[1]), 2),
                (np.linalg.norm(pts[1] - pts[2]), 0),
                (np.linalg.norm(pts[2] - pts[0]), 1)
            ]
            _, idx_opuesto = max(lados, key=lambda x: x[0])
            punto_clave = tuple(pts[idx_opuesto])
            puntos_referencia.append(punto_clave)
            cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)
            cv2.circle(output, punto_clave, 6, (0, 0, 255), -1)
        else:
            # opcional: si approx tiene 4 puntos muy pequeños (triángulo con ruido), podrías intentar detectar centro
            # vamos a ignorar otros shapes por ahora
            pass

    # eliminar duplicados cercanos (si dos puntos están a menos de 10 px son el mismo)
    if len(puntos_referencia) > 0:
        pts_np = np.array(puntos_referencia, dtype=np.float32)
        # cluster simple: eliminar puntos muy cercanos
        unique = []
        for p in pts_np:
            if not any(np.linalg.norm(p - q) < 10 for q in unique):
                unique.append(p)
        puntos_referencia = [tuple(p.astype(int)) for p in unique]

    vertices_rectangulo = None
    vertice_calculado = None

    if len(puntos_referencia) >= 3:
        if len(puntos_referencia) == 3:
            vertice_calculado, vertices_rectangulo = calcular_cuarto_vertice_mejorado(puntos_referencia)
            if vertices_rectangulo is not None:
                vertices_rectangulo = [v.astype(np.int32) for v in vertices_rectangulo]
                # dibujar vértice calculado
                cv2.circle(output, tuple(vertice_calculado.astype(int)), 10, (255, 0, 255), -1)
                cv2.putText(output, "CALC", (int(vertice_calculado[0]) + 10, int(vertice_calculado[1]) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            # si hay 4 o más, tomamos los 4 mejores (por ahora primeros 4 únicos)
            cuatro = puntos_referencia[:4]
            vertices_rectangulo = ordenar_vertices_por_posicion(cuatro).astype(np.int32)

    # Si tenemos rectángulo, dibujarlo y calcular escalas
    escala_horz = escala_vert = None
    if vertices_rectangulo is not None and len(vertices_rectangulo) == 4:
        v = np.array(vertices_rectangulo, dtype=np.float32)  # tl, tr, br, bl
        # dibujar
        cv2.polylines(output, [v.astype(np.int32)], isClosed=True, color=(0,255,255), thickness=3)
        for i,p in enumerate(v):
            cv2.circle(output, tuple(p.astype(int)), 6, (0,0,255), -1)
            cv2.putText(output, str(i+1), (int(p[0])+6, int(p[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # lados: top (0->1), right (1->2), bottom (2->3), left (3->0)
        top_len = np.linalg.norm(v[1] - v[0])
        right_len = np.linalg.norm(v[2] - v[1])
        bottom_len = np.linalg.norm(v[3] - v[2])
        left_len = np.linalg.norm(v[0] - v[3])

        avg_horz = (top_len + bottom_len) / 2.0
        avg_vert = (left_len + right_len) / 2.0

        # Asignar la dimensión A4 mayor (297) a la arista más larga en px
        if avg_horz >= avg_vert:
            # horizontales ~ 297 mm, verticales ~ 210 mm
            escala_horz = 297.0 / avg_horz   # mm por pixel para aristas horizontales (top/bottom)
            escala_vert = 210.0 / avg_vert   # mm por pixel para aristas verticales (left/right)
        else:
            # verticales ~ 297 mm, horizontales ~ 210 mm (hoja rotada)
            escala_horz = 210.0 / avg_horz
            escala_vert = 297.0 / avg_vert

        # Medir distancia mínima desde cada óvalo a cualquiera de las 4 aristas
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            # evitar triángulos
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                continue

            # obtener punto más bajo del contorno (como antes)
            idx = np.argmax(cnt[:, :, 1])
            x_max = int(cnt[idx][0][0])
            y_max = int(cnt[idx][0][1])
            p_ovalo = np.array([x_max, y_max], dtype=np.float32)

            # calcular distancia a cada arista (en px) y convertir a mm con la escala adecuada
            dist_px_list = []
            # aristas con su escala: top/bottom -> escala_horz ; right/left -> escala_vert
            edges = [
                (v[0], v[1], escala_horz),  # top
                (v[1], v[2], escala_vert),  # right
                (v[2], v[3], escala_horz),  # bottom
                (v[3], v[0], escala_vert),  # left
            ]

            for a, b, escala in edges:
                d_px = dist_punto_a_linea(p_ovalo, a, b)
                d_mm = d_px * escala
                dist_px_list.append(d_mm)

            # tomar la distancia mínima (el borde más cercano)
            dist_mm_min = float(np.min(dist_px_list))

            # dibujar y mostrar
            cv2.circle(output, (x_max, y_max), 5, (0, 0, 255), -1)
            cv2.putText(output, f"{dist_mm_min:.1f} mm", (x_max + 8, y_max - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Distancia al óvalo: {dist_mm_min:.2f} mm (min de {dist_px_list})")

    # Info de debug
    info_text = f"Triangulos detectados: {len(puntos_referencia)}"
    cv2.putText(output, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Thresh", thresh)
    cv2.imshow("Deteccion", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
