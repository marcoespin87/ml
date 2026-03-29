# 📊 Clasificador Automático de Reclamos Financieros

## Leveraging NLP & Machine Learning para la categorización inteligente de quejas de consumidores

---

## 📑 Índice de la Presentación

1. [Planteamiento del Problema](#1-planteamiento-del-problema)
2. [Uso de ML / DL](#2-uso-de-ml--dl)
3. [Datos y Preparación](#3-datos-y-preparación)
4. [Evaluación del Modelo](#4-evaluación-del-modelo)
5. [Implementación / Demo](#5-implementación--demo)

---

## 1. Planteamiento del Problema

### 🎯 Situación Actual

Las instituciones financieras reciben **cientos de miles de reclamos** de consumidores anualmente a través de la Consumer Financial Protection Bureau (CFPB). Cada reclamo contiene una narrativa en texto libre donde el consumidor describe su problema.

**El desafío crítico:** Estos reclamos llegan **sin categoría predefinida**. Un analista humano debe leer cada narrativa, interpretar el contenido y asignarla manualmente al departamento correspondiente (crédito, hipotecas, servicio al cliente, etc.).

### 📌 Problema Específico

- **Volumen:** El dataset contiene **555,957 reclamos** registrados, de los cuales **66,806** incluyen narrativas textuales detalladas.
- **Periodo:** Reclamos recibidos entre **marzo 2015** y **abril 2016** (~13 meses).
- **Costo operativo:** La clasificación manual es lenta, inconsistente y propensa a errores humanos.
- **Sin etiquetas temáticas:** Las narrativas no vienen pre-categorizadas por tipo de problema financiero — solo se registra el _producto_ (tarjeta, hipoteca, etc.) pero **no el motivo** del reclamo.

### 💡 Solución Propuesta

Desarrollar un **clasificador automático de reclamos** que:

1. **Descubra las categorías temáticas** ocultas en las narrativas usando aprendizaje no supervisado (Topic Modeling con NMF).
2. **Entrene modelos supervisados** con esas etiquetas descubiertas para clasificar automáticamente futuros reclamos.
3. **Despliegue el modelo** como una API REST consumible por un frontend web interactivo.

> **Impacto esperado:** Reducción del tiempo de triaje de reclamos de minutos a segundos, mejorando la experiencia del consumidor y optimizando los recursos operativos.

---

## 2. Uso de ML / DL

### 🏗️ Arquitectura Híbrida: Pipeline de Modelado en 2 Fases

El proyecto implementa una estrategia innovadora de **dos etapas** para resolver la ausencia de etiquetas iniciales:

```
┌──────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE MODELADO                          │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │  Texto Crudo │───▶│ NLP Pipeline │───▶│ TF-IDF (1000 dim)│    │
│  │  66,806 docs │    │ (spaCy)      │    │  Vectorización   │    │
│  └─────────────┘    └──────────────┘    └────────┬─────────┘    │
│                                                   │              │
│                          ┌────────────────────────┘              │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │         FASE 1: No Supervisada (NMF)                 │       │
│  │  • Topic Modeling → 4 clusters temáticos             │       │
│  │  • Validación por coherencia (c_v)                   │       │
│  │  • Genera etiquetas automáticas                      │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                          │ Etiquetas                             │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │        FASE 2: Supervisada (Clasificación)           │       │
│  │  • Random Forest        → 80% Accuracy               │       │
│  │  • Logistic Regression  → 80% Accuracy               │       │
│  │  • XGBoost ✅ (Ganador) → 83% Accuracy               │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

### Fase 1 — Aprendizaje No Supervisado: NMF (Non-Negative Matrix Factorization)

**¿Qué es NMF?** Es una técnica de factorización matricial que descompone la matriz documento-término (TF-IDF) en dos matrices de coeficientes no negativos, permitiendo descubrir tópicos latentes en los textos sin necesidad de etiquetas.

**¿Por qué NMF y no LDA?** NMF produce tópicos más interpretables en textos cortos/medianos como reclamos financieros, gracias a su restricción de no-negatividad que genera representaciones más limpias.

**Proceso de selección del número óptimo de clusters:**

- Se evaluaron **de 2 a 10 clusters** usando el puntaje de **coherencia c_v** (Gensim).
- Se seleccionaron **4 clusters** basándose en el primer pico de coherencia estable.

> 📊 **GRÁFICA:** Insertar aquí la gráfica de _"Coherence Score vs. Number of Clusters"_ (Cell 53 del notebook)

**Resultado:** 4 tópicos principales descubiertos, cada uno definido por sus 10 palabras más representativas.

### Fase 2 — Aprendizaje Supervisado: Clasificación Multiclase

Usando las etiquetas generadas por NMF, se entrenaron **3 modelos supervisados**:

| Modelo                  | Librería     | Hiperparámetros Clave                 |
| ----------------------- | ------------ | ------------------------------------- |
| **Random Forest**       | scikit-learn | `random_state=42` (default trees=100) |
| **Logistic Regression** | scikit-learn | `max_iter=1000`, `random_state=42`    |
| **XGBoost** ✅          | xgboost      | `eval_metric='mlogloss'`              |

**División de datos:** 80% entrenamiento / 20% prueba (`test_size=0.2`, `random_state=42`)

---

## 3. Datos y Preparación

### 📦 Dataset: Consumer Complaints (CFPB)

| Característica              | Valor                                       |
| --------------------------- | ------------------------------------------- |
| **Fuente**                  | Consumer Financial Protection Bureau (CFPB) |
| **Archivo**                 | `consumer_complaints.csv`                   |
| **Registros totales**       | 555,957                                     |
| **Registros con narrativa** | 66,806 (12%)                                |
| **Columnas**                | 18 atributos                                |
| **Periodo**                 | Marzo 2015 – Abril 2016                     |
| **Idioma**                  | Inglés                                      |

### Columnas Principales del Dataset

| Columna                        | Descripción                                    | Uso en el modelo               |
| ------------------------------ | ---------------------------------------------- | ------------------------------ |
| `consumer_complaint_narrative` | Texto libre del reclamo                        | **Input principal (features)** |
| `product`                      | Tipo de producto financiero (11 categorías)    | Análisis exploratorio          |
| `issue`                        | Tipo de problema reportado (95 categorías)     | Análisis exploratorio          |
| `state`                        | Estado del consumidor (62 estados/territorios) | Análisis geográfico            |
| `date_received`                | Fecha de recepción                             | Análisis temporal              |
| `company_response_to_consumer` | Respuesta de la empresa                        | Análisis exploratorio          |
| `timely_response`              | Si la respuesta fue oportuna                   | Análisis exploratorio          |

### Análisis Exploratorio de Datos (EDA)

> 📊 **GRÁFICA 1:** _"Complaints Over Time"_ — Tendencia mensual de reclamos (marzo 2015 – abril 2016). Muestra el volumen de quejas a lo largo del periodo. (Cell 17)

> 📊 **GRÁFICA 2:** _"Complaints by State"_ — Estados con más reclamos. California, Florida y New York lideran. (Cell 19)

> 📊 **GRÁFICA 3:** _"Complaints by Issue"_ — Top tipos de problemas reportados. (Cell 21)

> 📊 **GRÁFICA 4:** _"Complaints by Company Response"_ — Distribución tipo pie de las respuestas (cerrado con explicación, en progreso, etc.). (Cell 23)

> 📊 **GRÁFICA 5:** _"Complaints by Product"_ — Distribución por producto financiero (Mortgage, Debt collection, Credit reporting, etc.). (Cell 25)

> 📊 **GRÁFICA 6:** _"Timeliness of Company Response"_ — Proporción de respuestas oportunas vs. tardías. (Cell 26)

> 📊 **GRÁFICA 7:** _"Consumer Consent Provided"_ — Distribución tipo pie del consentimiento del consumidor. (Cell 27)

### 🔧 Pipeline de Ingeniería de Features (NLP)

La clave del éxito del modelo reside en el **filtrado semántico** del texto, no el volumen crudo. Se implementó un pipeline de 4 etapas usando **spaCy** (`en_core_web_sm`):

#### Etapa 1: Preprocesamiento Básico (`preprocess_text`)

```
Texto original:    "I called Chase Bank [XXXX] 3 times about my credit card #1234..."
Después:           "i called chase bank times about my credit card"
```

- Conversión a minúsculas
- Eliminación de texto entre corchetes `[...]`
- Eliminación de puntuación
- Eliminación de palabras con dígitos

#### Etapa 2: Lematización (`apply_lemmatization`)

```
Antes:  "called", "calling", "calls" → Después: "call"
Antes:  "reported", "reporting"      → Después: "report"
```

- Normalización a raíz base con spaCy
- Eliminación de stopwords y puntuación
- **Tiempo de procesamiento:** ~36 minutos para 66,806 documentos

#### Etapa 3: Filtrado de Sustantivos (`filter_nouns`)

```
Antes:  "call bank time service complaint"
Después: "bank time service complaint"  (solo sustantivos)
```

- POS Tagging (Part-of-Speech) con spaCy
- Se retienen **solo sustantivos** — eliminando el "ruido" de verbos y adjetivos
- El modelo se enfoca en los **conceptos clave** (tarjeta, cobro, reporte, hipoteca)
- **Tiempo de procesamiento:** ~36 minutos para 66,806 documentos

#### Etapa 4: Limpieza Final

- Eliminación de pronombres (`-PRON-`)
- Eliminación de tokens enmascarados (`xxxx`, `xx`) — credenciales ofuscadas del CFPB

### Visualización de Features

> 📊 **GRÁFICA 8:** _"Distribution of Complaint Length (Lemmatized)"_ — Histograma de la longitud de los textos procesados. (Cell 34)

> 📊 **GRÁFICA 9:** _"Top 50 Nouns in Complaints"_ — Nube de palabras ANTES de limpiar tokens enmascarados. Se observa `xxxx` como palabra dominante. (Cell 38)

> 📊 **GRÁFICA 10:** _"Top 50 Nouns in Complaints (limpio)"_ — Nube de palabras DESPUÉS de limpiar tokens. Las palabras clave reales emergen: credit, mortgage, bank, balance, etc. (Cell 47)

> 📊 **GRÁFICA 11:** _"Top 30 Unigrams in Complaints"_ — Las 30 palabras individuales más frecuentes. (Cell 41)

> 📊 **GRÁFICA 12:** _"Top 30 Bigrams in Complaints"_ — Las 30 combinaciones de 2 palabras más frecuentes. (Cell 42)

> 📊 **GRÁFICA 13:** _"Top 30 Trigrams in Complaints"_ — Las 30 combinaciones de 3 palabras más frecuentes. (Cell 43)

### Vectorización TF-IDF

Se transformó el texto limpio en representación numérica mediante **TF-IDF** (Term Frequency — Inverse Document Frequency):

| Parámetro                      | Valor       | Propósito                                               |
| ------------------------------ | ----------- | ------------------------------------------------------- |
| `stop_words`                   | `'english'` | Eliminar palabras comunes del inglés                    |
| `min_df`                       | `2`         | Descartar palabras que aparezcan en menos de 2 docs     |
| `max_df`                       | `0.95`      | Descartar palabras que aparezcan en más del 95% de docs |
| `max_features` (clasificación) | `1,000`     | Limitar dimensionalidad para clasificadores             |

---

## 4. Evaluación del Modelo

### 🏷️ Clasificación Multiclase: 4 Categorías Descubiertas

El modelo NMF descubrió **4 categorías temáticas** principales en las narrativas de reclamos:

| Cluster | Categoría                            | Top 10 Palabras Clave                                                                   | Total Docs   |
| ------- | ------------------------------------ | --------------------------------------------------------------------------------------- | ------------ |
| 0       | **Credit Reporting & Disputes**      | credit, bureaus, equifax, reported, information, report, remove, creditor, used, pulled | 3,959 (test) |
| 1       | **Customer Service & Communication** | called, time, did, bank, service, calling, stating, times, complaint, date              | 6,800 (test) |
| 2       | **Mortgages & Loan Servicing**       | mortgage, escrow, late, rate, modification, servicing, complaint, current, time, date   | 1,553 (test) |
| 3       | **Account Management & Fees**        | balance, paid, date, fees, transfer, chase, closed, late, reported, outstanding         | 1,050 (test) |

### 📊 Comparación de Modelos (Weighted F1-Score)

| Modelo              | Accuracy | Weighted F1-Score | Ranking      |
| ------------------- | -------- | ----------------- | ------------ |
| **XGBoost** ✅      | **83%**  | **0.8261**        | 🥇 **Mejor** |
| Random Forest       | 81%      | 0.8013            | 🥈           |
| Logistic Regression | 80%      | 0.7985            | 🥉           |

### Reporte de Clasificación — XGBoost (Modelo Ganador)

| Categoría                        | Precision | Recall   | F1-Score | Support    |
| -------------------------------- | --------- | -------- | -------- | ---------- |
| Credit Reporting & Disputes      | 0.83      | 0.82     | 0.82     | 3,959      |
| Customer Service & Communication | 0.86      | 0.85     | 0.85     | 6,800      |
| Mortgages & Loan Servicing       | 0.75      | 0.83     | 0.79     | 1,553      |
| Account Management & Fees        | 0.70      | 0.71     | 0.71     | 1,050      |
| **Weighted Avg**                 | **0.83**  | **0.83** | **0.83** | **13,362** |

### Interpretación de Resultados por Categoría

- **Customer Service (F1: 0.85):** La categoría con mejor rendimiento. Las quejas sobre atención al cliente tienen un vocabulario muy distintivo (called, calling, times, bank).
- **Credit Reporting (F1: 0.82):** Alta precisión gracias a términos específicos del dominio (equifax, bureaus, credit report).
- **Mortgages (F1: 0.79):** Buen recall (83%) — el modelo captura bien las hipotecas aunque algunas se confunden con otras categorías.
- **Account Management (F1: 0.71):** La categoría más desafiante. Tiene el menor volumen de datos (1,050) y su vocabulario se solapa con otras categorías (paid, fees, late).

### Visualización de la Evaluación

> 📊 **GRÁFICA 14:** _"Confusion Matrix — Random Forest"_ — Matriz de confusión para RF. (Cell 72)

> 📊 **GRÁFICA 15:** _"ROC Curves — Random Forest"_ — Curvas ROC multi-clase para RF con AUC por clase. (Cell 72)

> 📊 **GRÁFICA 16:** _"Confusion Matrix — Logistic Regression"_ — Matriz de confusión para LR. (Cell 73)

> 📊 **GRÁFICA 17:** _"ROC Curves — Logistic Regression"_ — Curvas ROC multi-clase para LR. (Cell 73)

> 📊 **GRÁFICA 18:** _"Confusion Matrix — XGBoost"_ — Matriz de confusión del modelo ganador. (Cell 74)

> 📊 **GRÁFICA 19:** _"ROC Curves — XGBoost"_ — Curvas ROC multi-clase para XGBoost. Esta gráfica demuestra la capacidad discriminativa superior del modelo. (Cell 74)

### Matriz de Confusión Detallada — XGBoost

```
                          Predicción
                    CRD     CSC     MLS     AMF
Actual CRD        3233     553      76      97
Actual CSC         527    5754     319     200
Actual MLS          59     174    1295      25
Actual AMF          75     190      35     750
```

- **CRD** = Credit Reporting & Disputes
- **CSC** = Customer Service & Communication
- **MLS** = Mortgages & Loan Servicing
- **AMF** = Account Management & Fees

**Insight:** La mayor confusión ocurre entre Credit Reporting ↔ Customer Service (553/527 errores cruzados), lo cual tiene sentido ya que los reclamos sobre reportes de crédito frecuentemente mencionan llamadas al servicio al cliente.

---

## 5. Implementación / Demo

### 🏛️ Arquitectura del Sistema

```
┌────────────────────┐    HTTP/REST     ┌──────────────────────┐
│                    │  ────────────▶   │                      │
│   Frontend (HTML)  │                  │  Backend (FastAPI)   │
│   • Formulario     │  ◀────────────   │  • /api/predict      │
│   • Resultados     │    JSON          │  • /api/categories   │
│   • Health check   │                  │  • /api/metrics      │
│                    │                  │  • /api/stats        │
└────────────────────┘                  │  • /api/health       │
                                        │                      │
                                        │  model.pkl (1.2 MB)  │
                                        └──────────────────────┘
```

### Backend — API REST (FastAPI)

**Ubicación:** `backend/server.py`

| Endpoint          | Método | Descripción                                                           |
| ----------------- | ------ | --------------------------------------------------------------------- |
| `/api/predict`    | POST   | Recibe texto del reclamo, retorna categoría predicha + probabilidades |
| `/api/categories` | GET    | Lista las 4 categorías disponibles                                    |
| `/api/metrics`    | GET    | Retorna métricas del modelo (classification report, confusion matrix) |
| `/api/stats`      | GET    | Estadísticas del dataset limpio                                       |
| `/api/health`     | GET    | Health check del servicio                                             |

**Artefactos del modelo:**

- `backend/models/model.pkl` (1.2 MB) → XGBoost + TF-IDF Vectorizer + LabelEncoder + Topic Mapping
- `backend/models/metrics.json` → Métricas de evaluación
- `backend/data/quejas_limpias.csv` → Dataset procesado para estadísticas

### Frontend — Interfaz Web

**Ubicación:** `frontend/index.html`

Interfaz web minimalista y elegante que permite:

- Escribir o pegar un reclamo en texto libre
- Enviar a la API para clasificación en tiempo real
- Visualizar la categoría predicha con nivel de confianza
- Ver Top 5 categorías con probabilidades

**Tecnologías:** HTML5 + Vanilla CSS + JavaScript (sin frameworks)

> 📸 **SCREENSHOT:** Insertar aquí una captura de pantalla del frontend con un ejemplo de clasificación exitosa.
