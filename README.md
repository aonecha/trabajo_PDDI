# Inferencia de Topología a partir de Señales sobre Grafos  
**Trabajo Final – Procesamiento de Datos Distribuidos e Inferencia (PDDI)**  
Universidad Rey Juan Carlos

---

## Descripción del proyecto

Este proyecto aborda el problema de la **inferencia de la topología de un grafo a partir de señales observadas en sus nodos**, en el marco del **Graph Signal Processing (GSP)** y los **modelos gráficos gaussianos**.

Concretamente, se estudia la inferencia de la estructura del grafo \( S \) utilizando **Graphical Lasso** y métodos de inferencia basados en la hipótesis de **señales estacionarias sobre grafos**. El análisis se realiza mediante datos sintéticos generados sobre distintos modelos de grafos.

---

## Objetivo

El objetivo principal es:

- **Inferir la estructura del grafo \( S \)** a partir de señales observadas en los nodos.
- Evaluar el rendimiento de distintos métodos de inferencia de topología.
- Analizar la sensibilidad de los métodos frente a distintos parámetros del problema.

---

## Generación de datos

### 🔹 Grafos sintéticos
Se generan grafos no dirigidos de tamaño aproximado \( N = 20 \) y \( N = 100 \) nodos, con un grado medio cercano a 4–6 enlaces por nodo, utilizando distintos modelos:

- Erdős–Rényi (ER)
- Small-World (SW)
- Barabási–Albert (BA)

---

### 🔹 Modelos de señales

Se consideran dos tipos de señales:

1. **Señales Gaussianas i.i.d.**
  x ~ N(0, S^{-1})

2. **Señales estacionarias sobre grafos**
   x = H w,   w ~ N(0, I)

   donde \( H \) es un filtro paso bajo definido sobre el grafo.

---

## Algoritmos e implementación

El proyecto pone énfasis en la **implementación de los algoritmos**, más que en su simple evaluación.

Se estudian y comparan distintas estrategias para la inferencia de grafos:

- **Graphical Lasso**
- Implementación mediante **CVXPY**
- Algoritmos iterativos clásicos:
  - Descenso por gradiente proyectado
  - Descenso coordinado para Graphical Lasso

El objetivo es comparar el impacto de distintas implementaciones sobre el rendimiento y el coste computacional.

---

## Métricas y análisis de sensibilidad

Para evaluar la calidad de la inferencia se utilizan las siguientes métricas:

- **Tiempo de cómputo** para la estimación del grafo.
- **Error de estimación del grafo**, definido como:

  Err(S, Ŝ) = || Ŝ − S ||_F / || S ||_F
  donde ||·||_F denota la norma de Frobenius.

---

### Análisis de sensibilidad

El rendimiento de los métodos se analiza en función de:

- Número de muestras disponibles.
- Número de nodos del grafo.
- Nivel de esparsidad del grafo.
- Tipo de grafo subyacente.
- Posible extensión a casos más realistas.

---

## Estructura del proyecto

  trabajo_PDDI/

  ├── data_generation.py

  ├── methods.py

  ├── metrics.py

  ├── experiments.py

  ├── experiments_big.py

  ├── plot_error_time_vs_M_all_graphs.py

  ├── requirements.txt

  ├── figures/

  └── figures_interpretation/

---

## Descripción de los archivos

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


## Reproducibilidad de los resultados

Esta sección describe los pasos necesarios para **reproducir todos los resultados del proyecto** a partir del repositorio.

1.  Instalar las dependencias:

pip install -r requirements.txt

2. Ejecución de los experimentos

a) Experimentos estándar (grafos pequeños)
Ejecuta todos los experimentos sobre grafos ER, Small-World y Barabási–Albert, considerando señales gaussianas y estacionarias, así como barridos en el número de muestras y el parámetro de regularización.

python experiments.py
Resultados obtenidos:

Métricas de error, esparsidad y tiempo por consola

Figuras de los grafos generados en figures/

b) Experimento a gran escala
Ejecuta un experimento más exigente para evaluar el rendimiento computacional de los métodos sobre un grafo grande.

python experiments_big.py
Configuración utilizada:

Grafo Erdős–Rényi

N = 100 nodos

M = 500 muestras

Los resultados se muestran por consola.

3. Generación de las figuras finales
Para reproducir todas las gráficas presentadas en el análisis (error, tiempo, esparsidad y frentes de Pareto), ejecutar:

python plot_error_time_vs_M_all_graphs.py

Figuras de los grafos generados en figures_interpretation/