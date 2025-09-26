# Detector de Distancias en Hoja A4

Este proyecto implementa un sistema de visión por computadora que detecta automáticamente los bordes de una hoja A4 usando triángulos de referencia y mide la distancia de objetos (óvalos) a los bordes de la hoja en milímetros reales.

## 📋 Descripción General

El sistema funciona detectando triángulos marcadores colocados en las esquinas de una hoja A4. A partir de estos marcadores, calcula automáticamente el cuarto vértice faltante (si solo hay 3 triángulos) y establece un sistema de referencia para convertir píxeles a milímetros, permitiendo mediciones precisas de distancias.

## 🔧 Dependencias

```python
import cv2
import numpy as np
from itertools import permutations
```

## 🎯 Características Principales

- **Detección automática de triángulos de referencia**
- **Cálculo inteligente del cuarto vértice faltante**
- **Calibración automática píxel-a-milímetro usando dimensiones A4**
- **Medición de distancias en tiempo real**
- **Interfaz interactiva con controles de umbralización**

---

## 📚 Documentación de Funciones

### `dist_punto_a_linea(p, a, b)`

Calcula la distancia perpendicular de un punto a una línea definida por dos puntos.

**Parámetros:**
- `p`: Punto del cual calcular la distancia (array numpy)
- `a`: Primer punto de la línea (array numpy)
- `b`: Segundo punto de la línea (array numpy)

**Retorna:**
- `float`: Distancia perpendicular en píxeles

**Funcionamiento:**
1. Convierte todos los puntos a arrays numpy float32
2. Calcula el vector director de la línea (b - a)
3. Usa el producto cruz para encontrar la distancia perpendicular
4. Aplica la fórmula: `|cross(b-a, p-a)| / |b-a|`

```python
# Ejemplo de uso
punto = [100, 150]
linea_inicio = [50, 100]
linea_fin = [200, 100]
distancia = dist_punto_a_linea(punto, linea_inicio, linea_fin)
# Resultado: distancia perpendicular en píxeles
```

---

### `ordenar_vertices_por_posicion(vertices)`

Ordena 4 vértices en el orden estándar: top-left, top-right, bottom-right, bottom-left.

**Parámetros:**
- `vertices`: Lista de 4 puntos [x, y]

**Retorna:**
- `numpy.array`: Vértices ordenados en sentido horario desde top-left

**Funcionamiento:**
1. Calcula el centroide de los 4 puntos
2. Determina el ángulo polar de cada vértice respecto al centro
3. Ordena por ángulo para obtener secuencia horaria
4. Rota la secuencia para que empiece en top-left (menor suma x+y)

```python
# Ejemplo de uso
vertices_desordenados = [[200, 300], [100, 100], [300, 100], [400, 300]]
vertices_ordenados = ordenar_vertices_por_posicion(vertices_desordenados)
# Resultado: [[100,100], [300,100], [400,300], [200,300]]  # TL, TR, BR, BL
```

---

### `calcular_cuarto_vertice_mejorado(puntos)`

Calcula el cuarto vértice de un rectángulo a partir de 3 puntos conocidos, probando todas las combinaciones posibles para encontrar el mejor rectángulo.

**Parámetros:**
- `puntos`: Lista de 3 puntos [x, y] conocidos

**Retorna:**
- `tuple`: (punto_calculado, vertices_rectangulo_completo)

**Funcionamiento:**
1. **Generación de candidatos**: Para cada permutación de 3 puntos (A, B, C):
   - Calcula D = A + C - B (propiedad del paralelogramo)
   - Verifica que D esté dentro de límites razonables

2. **Evaluación de calidad**: Para cada candidato:
   - Ordena los 4 vértices en posición estándar
   - Calcula un score de calidad usando `evaluar_rectangulo()`

3. **Selección del mejor**: Retorna el rectángulo con mayor score

```python
# Ejemplo de uso
tres_puntos = [[100, 100], [300, 120], [280, 300]]
cuarto_punto, rectangulo_completo = calcular_cuarto_vertice_mejorado(tres_puntos)
```

---

### `evaluar_rectangulo(vertices)`

Evalúa la calidad de un rectángulo basándose en múltiples criterios geométricos.

**Parámetros:**
- `vertices`: Array de 4 vértices ordenados

**Retorna:**
- `float`: Score de calidad (mayor = mejor, -1 = inválido)

**Criterios de evaluación:**

#### 1. **Ángulos próximos a 90°**
- Calcula cada ángulo interior usando producto punto
- Penaliza desviaciones mayores a 40° (retorna -1)
- Score = Σ(1 - desviación/40) para cada ángulo

#### 2. **Lados opuestos similares**
- Compara longitudes de lados opuestos (top vs bottom, left vs right)
- Penaliza diferencias mayores al 50% (retorna -1)
- Score += (1 - diferencia_relativa) para cada par

#### 3. **Proporción A4**
- Calcula ratio ancho/alto del rectángulo
- Compara con ratio A4 ideal (297/210 ≈ 1.414)
- Score += (1 - |ratio_actual - ratio_A4| / ratio_A4)

```python
# Ejemplo de evaluación
vertices = [[0,0], [210,0], [210,297], [0,297]]  # Rectángulo perfecto A4
score = evaluar_rectangulo(vertices)
# Score alto (cerca de 6.0 para rectángulo perfecto)
```

---

## 🔄 Flujo Principal del Programa

### 1. **Inicialización**
```python
# Carga de imagen
img = cv2.imread("hoja4.jpg")

# Configuración de ventanas
cv2.namedWindow("Deteccion", cv2.WINDOW_NORMAL)
cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)

# Control de umbralización
cv2.createTrackbar("Umbral", "Thresh", 23, 255, lambda x: None)
```

### 2. **Preprocesamiento de Imagen (Loop Principal)**

#### A. **Conversión y Filtrado**
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce ruido
```

#### B. **Umbralización Adaptiva**
```python
# Umbral base con Otsu
otsu_val, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Ajuste manual con trackbar
offset = umbral - 128  # Centrado en 0
thresh_val = max(0, min(255, otsu_val + offset))
_, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
```

#### C. **Limpieza Morfológica**
```python
kernel = np.ones((5,5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
# Une pequeños huecos en los objetos
```

### 3. **Detección de Triángulos de Referencia**

```python
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
puntos_referencia = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 150:  # Filtrar objetos muy pequeños
        continue
    
    # Aproximación poligonal
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    if len(approx) == 3:  # Es un triángulo
        # Encontrar vértice opuesto al lado más largo
        pts = approx.reshape(-1, 2)
        lados = [
            (np.linalg.norm(pts[0] - pts[1]), 2),  # lado 0-1, vértice opuesto: 2
            (np.linalg.norm(pts[1] - pts[2]), 0),  # lado 1-2, vértice opuesto: 0
            (np.linalg.norm(pts[2] - pts[0]), 1)   # lado 2-0, vértice opuesto: 1
        ]
        _, idx_opuesto = max(lados, key=lambda x: x[0])
        punto_clave = tuple(pts[idx_opuesto])
        puntos_referencia.append(punto_clave)
```

#### **¿Por qué el vértice opuesto al lado más largo?**
En triángulos marcadores colocados en esquinas de una hoja, el lado más largo suele ser la hipotenusa, y el vértice opuesto corresponde al ángulo recto que apunta hacia la esquina de la hoja.

### 4. **Eliminación de Duplicados**
```python
# Eliminar puntos muy cercanos (< 10 píxeles)
unique = []
for p in pts_np:
    if not any(np.linalg.norm(p - q) < 10 for q in unique):
        unique.append(p)
```

### 5. **Construcción del Rectángulo de Referencia**

#### **Caso 1: 3 triángulos detectados**
```python
if len(puntos_referencia) == 3:
    vertice_calculado, vertices_rectangulo = calcular_cuarto_vertice_mejorado(puntos_referencia)
```

#### **Caso 2: 4 o más triángulos**
```python
else:
    cuatro = puntos_referencia[:4]
    vertices_rectangulo = ordenar_vertices_por_posicion(cuatro)
```

### 6. **Calibración de Escalas**

Una vez establecido el rectángulo de referencia, se calculan las escalas de conversión píxel-a-milímetro:

```python
# Cálculo de longitudes promedio
v = vertices_rectangulo  # [tl, tr, br, bl]
top_len = np.linalg.norm(v[1] - v[0])
right_len = np.linalg.norm(v[2] - v[1])
bottom_len = np.linalg.norm(v[3] - v[2])
left_len = np.linalg.norm(v[0] - v[3])

avg_horz = (top_len + bottom_len) / 2.0    # Promedio aristas horizontales
avg_vert = (left_len + right_len) / 2.0    # Promedio aristas verticales

# Asignación inteligente de dimensiones A4
if avg_horz >= avg_vert:
    # Hoja en orientación landscape
    escala_horz = 297.0 / avg_horz  # mm/píxel para aristas horizontales
    escala_vert = 210.0 / avg_vert  # mm/píxel para aristas verticales
else:
    # Hoja en orientación portrait
    escala_horz = 210.0 / avg_horz
    escala_vert = 297.0 / avg_vert
```

### 7. **Medición de Distancias a Óvalos**

Para cada contorno que no sea un triángulo:

#### A. **Identificación del punto de medición**
```python
# Encontrar punto más bajo del óvalo (mayor coordenada Y)
idx = np.argmax(cnt[:, :, 1])
x_max = int(cnt[idx][0][0])
y_max = int(cnt[idx][0][1])
p_ovalo = np.array([x_max, y_max], dtype=np.float32)
```

#### B. **Cálculo de distancias a cada arista**
```python
edges = [
    (v[0], v[1], escala_horz),  # top edge
    (v[1], v[2], escala_vert),  # right edge
    (v[2], v[3], escala_horz),  # bottom edge
    (v[3], v[0], escala_vert),  # left edge
]

dist_mm_list = []
for a, b, escala in edges:
    d_px = dist_punto_a_linea(p_ovalo, a, b)  # Distancia en píxeles
    d_mm = d_px * escala                      # Conversión a milímetros
    dist_mm_list.append(d_mm)

# Tomar la distancia mínima (borde más cercano)
dist_mm_min = min(dist_mm_list)
```

---

## 🎛️ Controles de Usuario

### **Trackbar de Umbralización**
- **Rango**: 0-255
- **Valor inicial**: 23
- **Función**: Ajusta el umbral de binarización como offset sobre el valor Otsu
- **Uso**: Mueve el trackbar para optimizar la detección según las condiciones de iluminación

### **Teclas**
- **'q'**: Salir del programa

---

## 📊 Visualización

### **Ventana "Thresh"**
- Muestra la imagen binarizada después del procesamiento morfológico
- Útil para verificar que los objetos se detecten correctamente

### **Ventana "Deteccion"**
- **Triángulos detectados**: Contornos verdes con punto rojo en el vértice clave
- **Vértice calculado**: Círculo magenta con etiqueta "CALC"
- **Rectángulo de referencia**: Líneas cian con vértices numerados (1-4)
- **Óvalos**: Punto rojo en la posición de medición con distancia en texto verde
- **Información**: Contador de triángulos detectados en la esquina superior izquierda

---

## 🔧 Parámetros Ajustables

### **Filtros de Área**
```python
if area < 150:  # Triángulos mínimos
if cv2.contourArea(cnt) < 500:  # Óvalos mínimos
```

### **Aproximación Poligonal**
```python
approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 2% de tolerancia
```

### **Limpieza Morfológica**
```python
kernel = np.ones((5,5), np.uint8)  # Tamaño del kernel
iterations=1  # Número de iteraciones
```

### **Eliminación de Duplicados**
```python
if not any(np.linalg.norm(p - q) < 10 for q in unique):  # Distancia mínima 10px
```

---

## 🎯 Casos de Uso

1. **Control de calidad en impresión**: Verificar márgenes y posicionamiento de elementos
2. **Mediciones técnicas**: Distancias precisas en documentos físicos
3. **Calibración de sistemas**: Establecer referencias métricas en imágenes
4. **Análisis de documentos**: Medición automática de espaciados y posiciones

---

## ⚠️ Limitaciones y Consideraciones

### **Requisitos de la Imagen**
- Los triángulos deben ser claramente visibles y contrastados
- La hoja debe estar razonablemente plana (sin perspectiva extrema)
- Iluminación uniforme para mejor detección

### **Precisión**
- La precisión depende de la resolución de la imagen
- Distorsión de lente puede afectar mediciones en los bordes
- Se recomienda calibración con objetos de tamaño conocido para aplicaciones críticas

### **Robustez**
- El algoritmo maneja bien variaciones menores en la forma de los triángulos
- Puede fallar si hay muchos objetos ruidosos que interfieran con la detección
- Sensible a cambios extremos de iluminación

---

## 🚀 Posibles Mejoras

1. **Corrección de perspectiva**: Implementar transformación homográfica para corregir distorsiones
2. **Detección multi-escala**: Manejar objetos de diferentes tamaños automáticamente
3. **Filtros adaptativos**: Ajuste automático de parámetros según condiciones de imagen
4. **Exportación de resultados**: Guardar mediciones en archivos CSV o JSON
5. **Interfaz gráfica**: GUI más amigable para usuarios no técnicos

---

## 📝 Ejemplo de Salida

```
Distancia al óvalo: 15.32 mm (min de [23.45, 15.32, 45.67, 38.91])
Distancia al óvalo: 42.18 mm (min de [42.18, 56.23, 78.45, 91.34])
```

La salida muestra la distancia mínima al borde más cercano, junto con las distancias a todos los bordes para referencia.
