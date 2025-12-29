# Proyecto_PDDI

Este proyecto implementa un conjunto de **experimentos de aprendizaje de estructura en grafos** a partir de señales observadas, comparando distintos métodos de estimación de la **matriz de precisión (Θ)** y del **Laplaciano del grafo**.

Se estudia el compromiso entre **precisión**, **esparsidad** y **coste computacional** bajo diferentes tipos de grafos, modelos de señal y parámetros experimentales.

---

## 📁 Estructura del proyecto

.
├── data_generation.py
├── methods.py
├── metrics.py
├── experiments.py
├── experiments_big.py
├── plot_error_time_vs_M_all_graphs.py
├── requirements.txt
├── figures/
└── figures_interpretation/

yaml
Copiar código

---

## 📌 Descripción de los archivos

### `data_generation.py`
Contiene funciones para:

- Generación de grafos:
  - Erdős–Rényi (ER)
  - Watts–Strogatz (Small-World)
  - Barabási–Albert (Scale-Free)
- Generación de señales:
  - Modelo gaussiano basado en la matriz de precisión
  - Señales estacionarias mediante filtros laplacianos
- Cálculo de Laplacianos y matrices de precisión verdaderas

---

### `methods.py`
Implementa los métodos de inferencia de la matriz de precisión:

- **Ridge** (baseline mediante inversión regularizada)
- **Graphical Lasso (sklearn)**
- **Graphical Lasso (CVXPY)** con cacheo del problema
- **Projected Gradient Descent (PGD)**

Incluye también el cálculo de la covarianza muestral centrada.

---

### `metrics.py`
Define las métricas de evaluación:

- Error relativo de Frobenius (solo fuera de la diagonal)
- Error relativo de Frobenius completo (para Laplacianos)
- Esparsidad fuera de la diagonal
- Conversión de Θ estimada a Laplaciano
- (Opcional) F1-score del soporte del grafo

---

### `experiments.py`
Script principal de experimentación:

- Generación y visualización de grafos ER, SW y BA
- Ejecuciones individuales comparando métodos sobre el mismo grafo
- Barridos experimentales:
  - Error y tiempo vs número de muestras (M)
  - Error y esparsidad vs parámetro de regularización (λ)
- Soporte para señales gaussianas y estacionarias


### `Ejecución`
## 🔁 Reproducibilidad de los resultados

Esta sección describe los pasos necesarios para **reproducir todos los resultados del proyecto** a partir del repositorio.

1️⃣ Requisitos

- Python ≥ 3.10
- Sistema operativo: Linux / macOS / Windows
- Se recomienda el uso de un entorno virtual

Instalar las dependencias:

bash
pip install -r requirements.txt

2️⃣ Ejecución de los experimentos

a) Experimentos estándar (grafos pequeños)
Ejecuta todos los experimentos sobre grafos ER, Small-World y Barabási–Albert, considerando señales gaussianas y estacionarias, así como barridos en el número de muestras y el parámetro de regularización.

bash
Copiar código
python experiments.py
Resultados obtenidos:

Métricas de error, esparsidad y tiempo por consola

Figuras de los grafos generados en figures/

b) Experimento a gran escala
Ejecuta un experimento más exigente para evaluar el rendimiento computacional de los métodos sobre un grafo grande.

bash
Copiar código
python experiments_big.py
Configuración utilizada:

Grafo Erdős–Rényi

N = 100 nodos

M = 500 muestras

Los resultados se muestran por consola.

3️⃣ Generación de las figuras finales
Para reproducir todas las gráficas presentadas en el análisis (error, tiempo, esparsidad y frentes de Pareto), ejecutar:

bash
Copiar código
python plot_error_time_vs_M_all_graphs.py
Las figuras se generan automáticamente en el directorio:

Copiar código
figures_interpretation/