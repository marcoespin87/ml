# 🔬 Explicación Detallada del Notebook `ml.ipynb`
## Celda por celda: qué hace y por qué se hace

---

## 📦 FASE 0: Configuración del Entorno (Cells 0–4)

### Cell 0 — Instalación de spaCy
```python
!pip install spacy
```
**Qué hace:** Instala la librería spaCy, un framework de NLP (Natural Language Processing) industrial.

**Por qué:** spaCy es necesario para las etapas de lematización y POS tagging (filtrado de sustantivos) que se aplican más adelante. Se usa en lugar de NLTK porque es más rápido para procesamiento por lotes y tiene modelos pre-entrenados de alta calidad.

---

### Cell 1 — Descarga del modelo de idioma
```python
!python -m spacy download en_core_web_sm
```
**Qué hace:** Descarga el modelo de idioma inglés pequeño (`en_core_web_sm`, ~12 MB) de spaCy.

**Por qué:** Este modelo contiene las reglas lingüísticas del inglés (tokenización, lematización, POS tags, etc.) necesarias para procesar los textos de los reclamos. Se usa la versión "small" (`sm`) porque es suficiente para nuestro caso de uso y es mucho más rápida que los modelos `md` o `lg`.

---

### Cell 2 — Importación de módulos
```python
import numpy as np
import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
# ... más imports
```
**Qué hace:** Importa todas las librerías necesarias para el proyecto.

**Por qué cada una:**

| Librería | Propósito | Fase del pipeline |
|----------|-----------|-------------------|
| `numpy` | Operaciones numéricas con arrays | Todo el proyecto |
| `pandas` | Manipulación de DataFrames | Carga y exploración de datos |
| `re` | Expresiones regulares para limpieza de texto | Preprocesamiento NLP |
| `spacy` | Lematización y POS tagging | Ingeniería de features |
| `matplotlib` / `seaborn` | Visualizaciones y gráficas | EDA y evaluación |
| `wordcloud` | Generación de nubes de palabras | Visualización de features |
| `CountVectorizer` | Conteo de frecuencia de palabras (n-grams) | Análisis de vocabulario |
| `TfidfVectorizer` | Vectorización TF-IDF del texto | Transformación a features numéricas |
| `NMF` | Non-Negative Matrix Factorization | Topic Modeling (no supervisado) |
| `gensim` (Nmf, CoherenceModel) | NMF alternativo + cálculo de coherencia | Selección de clusters óptimo |
| `train_test_split` | División train/test | Preparación para clasificación |
| `RandomForestClassifier` | Modelo de ensamble basado en árboles | Clasificación supervisada |
| `LogisticRegression` | Modelo lineal | Clasificación supervisada |
| `XGBClassifier` | Gradient Boosting optimizado | Clasificación supervisada |
| `classification_report`, `confusion_matrix` | Métricas de evaluación | Evaluación del modelo |
| `roc_curve`, `auc` | Curvas ROC y área bajo la curva | Evaluación visual del modelo |
| `tqdm` | Barras de progreso | UX durante procesamiento largo |

---

### Cell 3 — Configuración global
```python
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None, 'display.max_columns', None, 'display.max_rows', None)
tqdm.pandas()
```
**Qué hace:**
1. Silencia warnings de deprecación que ensucian la salida
2. Configura pandas para mostrar todo el contenido sin truncar
3. Habilita `tqdm` para `pandas.apply()` con barra de progreso

**Por qué:** Mejora la experiencia durante el desarrollo — se ven los datos completos y no se interrumpe el flujo con warnings irrelevantes.

---

### Cell 4 — Inicialización de spaCy
```python
nlp = spacy.load('en_core_web_sm')
```
**Qué hace:** Carga el modelo de inglés en memoria como un objeto `nlp`.

**Por qué:** Este objeto se usará repetidamente en las funciones de lematización y filtrado de sustantivos. Cargarlo una vez es eficiente — evita recargar el modelo cada vez que se procesa un documento.

---

## 📂 FASE 1: Carga y Exploración de Datos (Cells 5–14)

### Cells 5–6 — Carga del CSV
```python
complaints_df = pd.read_csv("consumer_complaints.csv", low_memory=False)
complaints_df.head()
```
**Qué hace:** Lee el archivo CSV con 555,957 reclamos en un DataFrame de pandas.

**Por qué:** `low_memory=False` evita que pandas intente inferir tipos de datos por chunks, lo que puede causar errores con datasets grandes de tipos mixtos. El `.head()` es una inspección visual rápida para verificar que la carga fue correcta.

**Resultado:** DataFrame con 18 columnas y 555,957 filas.

---

### Cells 8–11 — Exploración del dataset
```python
complaints_df.info()              # Cell 8: Tipos de datos y valores nulos
complaints_df.describe()           # Cell 9: Estadísticas numéricas
complaints_df.describe(include='object').T  # Cell 10: Estadísticas de texto
complaints_df.columns             # Cell 11: Lista de columnas
```
**Qué hace:** Explora la estructura, tipos de datos, valores faltantes y estadísticas básicas del dataset.

**Por qué:** Es el paso estándar de **EDA (Exploratory Data Analysis)** — antes de modelar, necesitas entender:
- Cuántos **valores nulos** hay (la columna `consumer_complaint_narrative` solo tiene 66,806 de 555,957 — solo el 12% tiene texto)
- Cuántas **categorías únicas** existen (11 productos, 95 issues, 3,605 empresas)
- La **distribución** de los datos

**Hallazgo clave:** Solo **66,806 registros** (12%) tienen narrativa textual. El resto son reclamos sin descripción detallada, que no sirven para NLP.

---

### Cell 12 — Verificación de columnas
```python
for i, col in enumerate(complaints_df.columns):
    print(f"  {i}: {col}")
```
**Qué hace:** Lista todas las columnas con su índice.

**Por qué:** Al trabajar con un CSV externo (CFPB), es importante verificar que los nombres de las columnas son los esperados y no tienen prefijos o formatos inesperados. En este caso, las columnas están limpias (a diferencia del formato JSON original que tenía prefijos `_source.`).

---

### Cell 13 — Filtrado de registros con narrativa
```python
complaints_df = complaints_df[
    complaints_df['consumer_complaint_narrative'].replace('', np.nan).notnull()
]
```
**Qué hace:** Elimina todas las filas donde `consumer_complaint_narrative` es vacía o NaN.

**Por qué:** **Este es un paso crítico.** El modelo se basa en texto — sin narrativa, no hay features para el modelo. Se reducen de 555,957 a **66,806 registros** útiles. El `.replace('', np.nan)` es una precaución adicional: convierte strings vacías a NaN antes de filtrar, por si hay celdas que parecen tener datos pero están en blanco.

---

### Cell 14 — Verificación post-filtrado
```python
complaints_df.info()
```
**Qué hace:** Confirma que el filtrado funcionó correctamente.

**Resultado:** 66,806 registros, todos con narrativa no nula. Memory: 9.7 MB (reducido de 76.3 MB).

---

## 📊 FASE 2: Análisis Exploratorio Visual (Cells 15–27)

### Cells 16–17 — Reclamos en el tiempo
```python
complaints_df['date_received'] = pd.to_datetime(complaints_df['date_received'], format='%m/%d/%Y')
monthly_complaints = complaints_df.set_index('date_received').resample('ME').size()
```
**Qué hace:** Convierte las fechas a formato datetime y agrupa los reclamos por mes para crear una serie temporal.

**Por qué:** Visualizar la **tendencia temporal** revela patrones estacionales o picos de actividad. Si hay un mes con muchos más reclamos, podría indicar un evento externo (crisis financiera, cambio regulatorio, etc.).

**Resultado:** Los datos van de **marzo 2015 a abril 2016** (~13 meses).

> 📊 Se genera la gráfica: _"Complaints Over Time"_

---

### Cell 19 — Reclamos por estado
```python
state_complaints = complaints_df['state'].value_counts()
```
**Qué hace:** Cuenta los reclamos por estado geográfico.

**Por qué:** Identifica las **regiones con mayor actividad de quejas**. Esto puede correlacionar con la densidad poblacional o con prácticas específicas de empresas en ciertos estados.

> 📊 Se genera la gráfica: _"Complaints by State"_ — California, Florida y New York lideran.

---

### Cell 21 — Reclamos por tipo de problema (Issue)
```python
issue_complaints = complaints_df['issue'].value_counts()
```
**Qué hace:** Desglosa los 95 tipos de problemas reportados.

**Por qué:** Muestra qué problemas son más frecuentes — información valiosa para priorizar recursos de atención al cliente.

> 📊 Se genera la gráfica: _"Complaints by Issue"_ (gráfica horizontal)

---

### Cell 23 — Respuesta de la empresa
```python
company_response_counts = complaints_df['company_response_to_consumer'].value_counts()
```
**Qué hace:** Analiza cómo las empresas responden a los reclamos.

**Por qué:** Evalúa la calidad de respuesta del ecosistema financiero. Las categorías incluyen: "Closed with explanation", "Closed with monetary relief", "In progress", etc.

> 📊 Se genera la gráfica tipo pie: _"Complaints by Company Response"_

---

### Cell 25 — Reclamos por producto financiero
```python
product_complaints = complaints_df['product'].value_counts()
```
**Qué hace:** Distribuye los reclamos por los 11 productos financieros.

**Por qué:** Revela qué **productos generan más quejas** — Mortgage, Debt collection y Credit reporting suelen liderar. Esto da contexto para entender los clusters que NMF descubrirá después.

> 📊 Se genera la gráfica: _"Complaints by Product"_

---

### Cell 26 — Oportunidad de respuesta
```python
timely_counts = complaints_df['timely_response'].value_counts()
```
**Qué hace:** Muestra si las empresas respondieron a tiempo (Yes/No).

**Por qué:** Indicador de cumplimiento regulatorio — la CFPB exige respuestas oportunas.

> 📊 Se genera la gráfica: _"Timeliness of Company Response"_

---

### Cell 27 — Consentimiento del consumidor
```python
consent_counts = complaints_df['consumer_consent_provided'].value_counts()
```
**Qué hace:** Muestra la distribución del consentimiento de publicación.

**Por qué:** Solo los reclamos con consentimiento tienen narrativa pública — explica por qué solo 66,806 de 555,957 tienen texto.

> 📊 Se genera la gráfica tipo pie: _"Consumer Consent Provided"_

---

## 🧹 FASE 3: Preprocesamiento NLP (Cells 28–47)

### Cell 29 — Definición de funciones de preprocesamiento

#### Función 1: `preprocess_text(text)`
```python
def preprocess_text(text):
    text = text.lower()                          # Minúsculas
    text = re.sub(r'\[.*?]', ' ', text)          # Eliminar [contenido entre corchetes]
    text = re.sub(r'[^\w\s]', ' ', text)         # Eliminar puntuación
    text = re.sub(r'\w*\d\w*', ' ', text)        # Eliminar palabras con números
    text = re.sub(r'\s+', ' ', text)             # Colapsar espacios múltiples
    return text.strip()
```
**Qué hace:** Limpia el texto crudo eliminando ruido.

**Por qué cada paso:**
1. **Minúsculas:** "Credit" y "credit" deben ser la misma palabra para el modelo
2. **Corchetes:** El CFPB enmascara datos sensibles como `[XXXX Bank]` — esto agrega ruido
3. **Puntuación:** Signos como `,`, `.`, `!` no aportan significado semántico para clasificación de temas
4. **Números:** Tokens como fechas (`03/15/2016`), montos (`$1234`) o IDs de cuenta no son útiles para determinar el *tipo* de reclamo
5. **Espacios:** Las eliminaciones previas dejan espacios dobles que hay que limpiar

---

#### Función 2: `apply_lemmatization(text_list)`
```python
def apply_lemmatization(text_list):
    lemmatized_texts = []
    for doc in tqdm(nlp.pipe(text_list), total=len(text_list)):
        lemmatized_texts.append(
            ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
        )
    return lemmatized_texts
```
**Qué hace:** Convierte cada palabra a su **forma base (lema)** y elimina stopwords.

**Por qué:**
- `"called"`, `"calling"`, `"calls"` → `"call"` — reduce la dimensionalidad del vocabulario
- `"reported"`, `"reporting"`, `"reports"` → `"report"` — consolida variantes de la misma palabra
- Stopwords como "the", "is", "and" se eliminan porque no aportan significado temático
- `nlp.pipe()` procesa en batch, mucho más eficiente que `nlp()` documento por documento

**Tiempo:** ~36 minutos para 66,806 documentos (31 docs/segundo)

---

#### Función 3: `filter_nouns(text_list)`
```python
def filter_nouns(text_list):
    noun_texts = []
    for doc in tqdm(nlp.pipe(text_list), total=len(text_list)):
        noun_texts.append(' '.join(token.text for token in doc if token.pos_ == 'NOUN'))
    return noun_texts
```
**Qué hace:** Retiene **únicamente los sustantivos** del texto usando POS Tagging (Part-of-Speech).

**Por qué:** Este es el paso más ingenioso del pipeline. Los sustantivos capturan los **conceptos clave** del reclamo:
- ✅ Sustantivos útiles: `credit`, `mortgage`, `bank`, `balance`, `fees`, `escrow`
- ❌ Verbos eliminados: `called`, `said`, `told`, `went` — no distinguen temas
- ❌ Adjetivos eliminados: `good`, `bad`, `late` — demasiado genéricos

**Beneficio clave:** Al reducir el texto solo a sustantivos, el modelo de tópicos (NMF) puede enfocarse en los conceptos financieros que realmente diferencian un tipo de reclamo de otro.

**Tiempo:** ~36 minutos para 66,806 documentos

---

### Cell 30 — Aplicar preprocesamiento básico
```python
processed_df['complaint_processed'] = [
    preprocess_text(str(text)) for text in tqdm(complaints_df['consumer_complaint_narrative'])
]
```
**Qué hace:** Aplica la función `preprocess_text` a las 66,806 narrativas.

**Tiempo:** ~8 segundos (8,085 textos/seg) — es rápido porque solo usa regex, no modelos de ML.

---

### Cell 31 — Aplicar lematización
```python
processed_df['complaint_lemmatized'] = apply_lemmatization(processed_df['complaint_processed'])
```
**Qué hace:** Lematiza todos los textos preprocesados.

**Tiempo:** ~36 minutos — es lento porque usa el modelo de spaCy para cada token.

---

### Cell 32 — Filtrar sustantivos
```python
processed_df['complaint_nouns'] = filter_nouns(processed_df['complaint_lemmatized'])
```
**Qué hace:** Extrae solo los sustantivos de los textos lematizados.

**Tiempo:** ~36 minutos — misma razón que la lematización.

---

### Cell 34 — Distribución de longitud de textos
```python
complaint_length = processed_df['complaint_lemmatized'].str.len()
complaint_length.plot(kind='hist', bins=40)
```
**Qué hace:** Grafica un histograma de la longitud de los textos procesados.

**Por qué:** Verificar que no hay textos excesivamente cortos (que aporten poco) ni excesivamente largos (que dominen el TF-IDF). También revela la distribución típica — la mayoría de reclamos tienen entre 200–1000 caracteres lematizados.

> 📊 Se genera: _"Distribution of Complaint Length — Lemmatized"_

---

### Cells 36–38 — Word Cloud (antes de limpiar)
```python
all_nouns = ' '.join(processed_df['complaint_nouns'])
wordcloud = WordCloud(max_words=50, ...).generate(all_nouns)
```
**Qué hace:** Genera una nube de las 50 palabras (sustantivos) más frecuentes.

**Por qué:** Inspección visual rápida del vocabulario. En esta primera nube, **`xxxx` domina el gráfico** — son tokens de ofuscación del CFPB que enmascaran datos sensibles (números de tarjeta, nombres, etc.). Esto señala la necesidad del siguiente paso de limpieza.

> 📊 Se genera: _"Top 50 Nouns in Complaints"_ (con `xxxx`)

---

### Cell 39 — Limpieza de pronombres
```python
processed_df['complaint_cleaned'] = processed_df['complaint_nouns'].apply(lambda x: x.replace('-PRON-', ''))
```
**Qué hace:** Elimina el token `-PRON-` que spaCy inserta como lema de pronombres.

**Por qué:** spaCy reemplaza pronombres (I, he, she, they) con `-PRON-` durante la lematización. Este token no aporta información temática.

---

### Cells 41–43 — Análisis de N-grams
```python
# Unigrams (Cell 41)
get_top_n_words([all_complaints], 30)

# Bigrams (Cell 42)
get_top_n_bigram([all_complaints], 30)

# Trigrams (Cell 43)
get_top_n_trigram([all_complaints], 30)
```
**Qué hace:** Calcula las 30 palabras individuales, pares y tripletes más frecuentes.

**Por qué:**
- **Unigrams:** Revelan las palabras clave dominantes (credit, mortgage, bank, balance)
- **Bigrams:** Muestran frases compuestas relevantes (credit report, bank account, customer service)
- **Trigrams:** Capturan patrones más largos (credit bureau equifax, bank account service)

Estos análisis validan que el preprocesamiento NLP está funcionando y que las palabras más relevantes del dominio financiero emergen naturalmente.

> 📊 Se generan: _"Top 30 Unigrams"_, _"Top 30 Bigrams"_, _"Top 30 Trigrams"_

---

### Cells 45–47 — Limpieza final de tokens enmascarados
```python
processed_df['complaint_cleaned'] = processed_df['complaint_cleaned'].str.replace('xxxx', '')
processed_df['complaint_cleaned'] = processed_df['complaint_cleaned'].str.replace('xx', '')
```
**Qué hace:** Elimina los tokens de ofuscación `xxxx` y `xx` del CFPB.

**Por qué:** Estos tokens representan datos censurados (números de tarjeta, direcciones, etc.) y eran las "palabras" más frecuentes, distorsionando el análisis. Después de limpiarlos, la nube de palabras muestra los conceptos financieros reales.

> 📊 Se genera: _"Top 50 Nouns in Complaints"_ (limpia, sin `xxxx`)

---

## 🔢 FASE 4: Vectorización TF-IDF (Cells 48–50)

### Cell 49 — Creación de la matriz TF-IDF
```python
tfidf_vect = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.95)
X = tfidf_vect.fit_transform(processed_df['complaint_cleaned'])
```
**Qué hace:** Transforma los textos limpios en una **matriz numérica** donde cada fila es un documento y cada columna es una palabra, ponderada por TF-IDF.

**¿Qué es TF-IDF?**
- **TF (Term Frequency):** Cuántas veces aparece una palabra en un documento
- **IDF (Inverse Document Frequency):** Penaliza palabras que aparecen en muchos documentos (son menos discriminativas)
- **TF-IDF = TF × IDF:** Palabras frecuentes en pocos documentos tienen peso alto; palabras comunes en todos los documentos tienen peso bajo

**Por qué los parámetros:**

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `stop_words='english'` | Eliminar stopwords | Redundancia con el filtro de spaCy, pero agrega protección extra |
| `min_df=2` | Mínimo 2 documentos | Elimina palabras ultra-raras (probablemente errores tipográficos o nombres propios irrelevantes) |
| `max_df=0.95` | Máximo 95% de documentos | Elimina palabras que aparecen en casi todos los docs (no discriminan temas) |

**Resultado:** Se crea la **Document-Term Matrix (DTM)** que es la entrada para NMF.

---

### Cell 50 — Inspección de la DTM
```python
dtm_df = pd.DataFrame(X.toarray(), columns=tfidf_vect.get_feature_names_out())
```
**Qué hace:** Convierte la matriz dispersa a un DataFrame para inspección visual.

**Por qué:** Verificar que la matriz tiene sentido — la mayoría de celdas son 0.0 (como es esperado en una matriz dispersa de texto), y los valores no-cero representan la relevancia de cada palabra en cada documento.

---

## 🧩 FASE 5: Topic Modeling con NMF (Cells 51–63)

### Cell 53 — Búsqueda del número óptimo de clusters
```python
coherence_scores = []
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    nmf_model_gensim = Nmf(corpus=corpus_gensim, num_topics=n_clusters, id2word=dictionary, passes=5)
    cm = CoherenceModel(model=nmf_model_gensim, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_scores.append(cm.get_coherence())
```
**Qué hace:** Entrena modelos NMF con diferente número de clusters (2 a 10) y calcula el puntaje de **coherencia c_v** para cada uno.

**¿Qué es la coherencia c_v?** Mide qué tan "coherentes" son los tópicos descubiertos — es decir, si las palabras dentro de un mismo tópico tienden a aparecer juntas en los documentos. Un puntaje más alto = tópicos más interpretables.

**Por qué este enfoque:** No hay una "verdad absoluta" para el número de tópicos — se usa la coherencia como criterio objetivo para elegir el punto donde los tópicos son más significativos sin ser redundantes.

**¿Por qué Gensim y no sklearn para este paso?** sklearn no tiene coherencia built-in. Gensim sí implementa `CoherenceModel` con múltiples métricas (`c_v`, `u_mass`, etc.). Se usa Gensim *solo* para la selección del número óptimo, y luego sklearn para el modelo final.

> 📊 Se genera: _"Coherence Score vs. Number of Clusters"_

**Resultado:** Se seleccionaron **4 clusters** como el número óptimo.

---

### Cell 56 — Entrenamiento del modelo NMF final
```python
best_num_clusters = 4
nmf_model = NMF(n_components=best_num_clusters, random_state=42)
nmf_features = nmf_model.fit_transform(X)
```
**Qué hace:** Entrena el modelo NMF de sklearn con 4 componentes (tópicos) usando la matriz TF-IDF.

**¿Cómo funciona NMF internamente?**
```
X (66,806 × vocab)  ≈  W (66,806 × 4)  ×  H (4 × vocab)
   (documentos)        (doc→tópico)        (tópico→palabras)
```
- **W:** Cada fila dice "cuánto pertenece este documento a cada tópico"
- **H:** Cada fila dice "cuáles son las palabras más importantes de este tópico"

**`random_state=42`:** Garantiza reproducibilidad — NMF tiene inicialización aleatoria.

---

### Cell 58 — Inspección de clusters
```python
for i in range(best_num_clusters):
    top_words = get_top_words_in_cluster(i, feature_names)
    print(f"Cluster {i+1}: {top_words}")
```
**Qué hace:** Muestra las 10 palabras más representativas de cada cluster.

**Por qué:** Es el paso de **validación humana** — un experto revisa las palabras para verificar que los tópicos tienen sentido semántico y los nombra.

**Resultados:**
```
Cluster 1: ['credit', 'bureaus', 'equifax', 'reported', 'information', 'report', 'remove', 'creditor', 'used', 'pulled']
Cluster 2: ['called', 'time', 'did', 'bank', 'service', 'calling', 'stating', 'times', 'complaint', 'date']
Cluster 3: ['mortgage', 'escrow', 'late', 'rate', 'time', 'modification', 'date', 'servicing', 'complaint', 'current']
Cluster 4: ['balance', 'paid', 'date', 'fees', 'transfer', 'chase', 'closed', 'late', 'reported', 'outstanding']
```

**Interpretación:**
- **Cluster 1** → Problemas con reportes de crédito y burós → *Credit Reporting & Disputes*
- **Cluster 2** → Quejas sobre atención telefónica → *Customer Service & Communication*
- **Cluster 3** → Hipotecas, escrow, modificaciones → *Mortgages & Loan Servicing*
- **Cluster 4** → Balances, comisiones, cuentas cerradas → *Account Management & Fees*

---

### Cells 60–62 — Asignación de clusters y mapeo de tópicos
```python
processed_df['Cluster'] = nmf_model.transform(X).argmax(axis=1)

topic_mapping = {
    0: 'Credit Reporting & Disputes',
    1: 'Customer Service & Communication',
    2: 'Mortgages & Loan Servicing',
    3: 'Account Management & Fees'
}

processed_df['Topic'] = processed_df['Cluster'].map(topic_mapping)
```
**Qué hace:**
1. Cada documento recibe el cluster al que más pertenece (`argmax` del vector W)
2. Se asignan nombres legibles a cada cluster
3. Se crea la columna 'Topic' con los nombres

**Por qué:** Estas etiquetas de tópicos son las **pseudo-etiquetas** que se usarán como target (`y`) para los modelos supervisados. NMF actúa como un "generador de etiquetas" — resolviendo el problema de no tener categorías predefinidas.

---

## 🤖 FASE 6: Clasificación Supervisada (Cells 64–74)

### Cell 65–68 — Preparación de datos para clasificación
```python
data = processed_df[['complaint_processed', 'Cluster']]
X = data['complaint_processed']
y = data['Cluster']
y = pd.Categorical(y).codes
```
**Qué hace:**
1. Selecciona solo el texto procesado y el cluster como columnas
2. Define X (features = texto) e y (target = cluster)
3. Convierte los clusters a códigos numéricos (0, 1, 2, 3)

**Por qué se usa `complaint_processed` y NO `complaint_cleaned`:** El texto para clasificación usa el preprocesamiento básico (minúsculas, sin puntuación), no el filtrado de sustantivos. Esto es porque los **clasificadores supervisados** pueden beneficiarse de más contexto textual que el modelo no supervisado.

---

### Cell 69 — División Train/Test
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
**Qué hace:** Divide los datos en 80% entrenamiento y 20% prueba.

**Por qué:**
- **80/20** es el estándar de la industria para datasets de este tamaño
- **`random_state=42`:** Garantiza que la división es reproducible
- **Resultado:** Train = 53,444 docs, Test = 13,362 docs

---

### Cell 70 — Vectorización para clasificación
```python
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
```
**Qué hace:** Crea una nueva vectorización TF-IDF con solo las 1,000 palabras más relevantes.

**Por qué `max_features=1000`:** Limitar las dimensiones tiene 3 beneficios:
1. **Velocidad:** Menos features = entrenamiento más rápido
2. **Generalización:** Evita overfitting por tener demasiadas features sparse
3. **Memoria:** Reduce el uso de RAM significativamente

**¿Por qué un vectorizer diferente al de NMF?** El de NMF usó `min_df=2, max_df=0.95` sin límite de features (vocabulario completo). Para clasificación supervisada, 1,000 features son suficientes y más eficientes.

**IMPORTANTE:** Se hace `fit_transform` solo en train y `transform` en test — **nunca se fitea en test** para evitar data leakage.

---

### Cell 71 — Funciones de evaluación
```python
def plot_confusion_matrix(cm, classes, ...):
    # Visualiza la matriz de confusión con colores

def train_eval_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    # Genera confusion matrix
    # Genera ROC curves
```
**Qué hace:** Define funciones reutilizables para entrenar, evaluar y visualizar cada modelo.

**Por qué se definen como funciones:** Se van a evaluar 3 modelos — tener funciones evita duplicar código y garantiza que la evaluación es consistente entre modelos.

**Métricas calculadas:**
- **Classification Report:** Precision, Recall, F1-Score por clase
- **Confusion Matrix:** Muestra dónde el modelo acierta y se confunde
- **ROC Curves:** Curvas ROC multi-clase con AUC (Area Under Curve) por cada categoría

---

### Cell 72 — Random Forest
```python
rf_model = RandomForestClassifier(random_state=42)
train_eval_model(rf_model, "Random Forest", X_train, y_train, X_test, y_test)
```
**Qué hace:** Entrena y evalúa un Random Forest con 100 árboles (default).

**¿Qué es Random Forest?** Ensamble de árboles de decisión que votan. Cada árbol se entrena con un subconjunto aleatorio de datos y features.

**Resultado:** **81% Accuracy**, F1 weighted: 0.80. Punto débil: Account Management (F1: 0.59).

> 📊 Se generan: Confusion Matrix + ROC Curves para RF

---

### Cell 73 — Logistic Regression
```python
lr_model = LogisticRegression(random_state=42, max_iter=1000)
train_eval_model(lr_model, "Logistic Regression", X_train, y_train, X_test, y_test)
```
**Qué hace:** Entrena y evalúa una Regresión Logística multiclase.

**¿Qué es Logistic Regression?** Modelo lineal que calcula la probabilidad de pertenencia a cada clase. `max_iter=1000` asegura convergencia (el default de 100 puede no ser suficiente con 1,000 features).

**Resultado:** **80% Accuracy**, F1 weighted: 0.80. Similar a RF pero ligeramente inferior.

> 📊 Se generan: Confusion Matrix + ROC Curves para LR

---

### Cell 74 — XGBoost
```python
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
train_eval_model(xgb_model, "XGBoost", X_train, y_train, X_test, y_test)
```
**Qué hace:** Entrena y evalúa un XGBoost (eXtreme Gradient Boosting).

**¿Qué es XGBoost?** Algoritmo de gradient boosting optimizado que construye árboles secuencialmente, donde cada nuevo árbol corrige los errores del anterior. Es conocido por ganar competencias de ML.

**`use_label_encoder=False`:** Evita un warning de deprecación.
**`eval_metric='mlogloss'`:** Usa logarithmic loss para multiclase.

**Resultado:** **83% Accuracy**, F1 weighted: 0.83. **Mejor modelo.** Destaca especialmente en Account Management (F1: 0.71 vs 0.59 de RF).

> 📊 Se generan: Confusion Matrix + ROC Curves para XGBoost

---

## 🏆 FASE 7: Comparación y Guardado (Cells 75–78)

### Cell 76 — Comparación de modelos
```python
results = {
    'Random Forest': rf_report['weighted avg']['f1-score'],       # 0.8013
    'Logistic Regression': lr_report['weighted avg']['f1-score'],  # 0.7985
    'XGBoost': xgb_report['weighted avg']['f1-score'],             # 0.8261
}
best_model_name = max(results, key=results.get)  # → XGBoost
```
**Qué hace:** Compara los 3 modelos por F1-Score ponderado y selecciona al ganador.

**¿Por qué F1 y no solo Accuracy?** El F1-Score balancea Precision y Recall. Con clases desbalanceadas (Customer Service tiene 6,800 vs Account Management con 1,050), accuracy puede ser engañoso — un modelo que siempre prediga la clase mayoritaria tendría ~50% de accuracy sin ser útil.

**Resultado:** XGBoost gana con **F1 weighted = 0.8261**.

---

### Cell 78 — Guardado de artefactos
```python
artifacts = {
    'model': best_model,
    'tfidf': vectorizer,
    'target_encoder': label_encoder,
    'topic_mapping': topic_mapping,
    'categories': categories,
    'best_model_name': best_model_name,
}
joblib.dump(artifacts, MODEL_PATH)
```
**Qué hace:** Guarda el modelo ganador y todos los componentes necesarios para inferencia en un archivo `.pkl`.

**Por qué se guardan todos juntos:**
- **`model`:** El clasificador XGBoost entrenado
- **`tfidf`:** El vectorizador TF-IDF (necesario para transformar textos nuevos al mismo espacio vectorial)
- **`target_encoder`:** Mapea los códigos numéricos (0,1,2,3) a los nombres de categorías
- **`topic_mapping`:** Diccionario cluster→nombre del tópico
- **`categories`:** Lista ordenada de nombres de categorías

**Guardar todo en un solo pickle** simplifica el despliegue — el backend solo necesita cargar un archivo para tener todo listo para predecir.

**También se guardan:**
- `metrics.json` — Para el endpoint `/api/metrics` del backend
- `quejas_limpias.csv` — Para el endpoint `/api/stats` del backend

---

## 📊 Resumen del Pipeline Completo

```
    Texto crudo (66,806 narrativas)
              │
              ▼
    ┌─── Preprocesamiento ───┐
    │  • Minúsculas           │
    │  • Eliminar [...]       │
    │  • Eliminar puntuación  │
    │  • Eliminar números     │
    └─────────┬───────────────┘
              ▼
    ┌─── Lematización (spaCy) ──┐
    │  "called" → "call"        │
    │  "reporting" → "report"   │
    │  − Eliminar stopwords     │
    └─────────┬─────────────────┘
              ▼
    ┌─── Filtrado Sustantivos ──┐
    │  POS Tagging: solo NOUN   │
    │  "credit bank balance"    │
    └─────────┬─────────────────┘
              ▼
    ┌─── Limpieza Final ────────┐
    │  − Eliminar -PRON-        │
    │  − Eliminar xxxx, xx      │
    └─────────┬─────────────────┘
              ▼
    ┌─── TF-IDF Vectorización ──┐
    │  Texto → Matriz numérica  │
    └─────────┬─────────────────┘
              ▼
    ┌─── NMF (No Supervisado) ──┐
    │  → 4 tópicos descubiertos │
    │  → Genera etiquetas       │
    └─────────┬─────────────────┘
              ▼
    ┌─── Train/Test Split ──────┐
    │  80% train / 20% test     │
    └─────────┬─────────────────┘
              ▼
    ┌─── Clasificación ─────────┐
    │  RF: 81% | LR: 80%       │
    │  XGBoost: 83% ✅          │
    └─────────┬─────────────────┘
              ▼
    ┌─── Guardado ──────────────┐
    │  model.pkl (1.2 MB)       │
    │  metrics.json             │
    │  → Backend API (FastAPI)  │
    └───────────────────────────┘
```
