# 🎵 Predicción de Popularidad de Canciones (Spotify Audio Features)
- **Felipe Agudelo**
- **Juan Manuel Lopera**

Este proyecto desarrolla un sistema de aprendizaje automático para predecir si una canción será **popular** en Spotify a partir de sus **características de audio**. Se implementaron dos enfoques de modelado: **Random Forest** y **XGBoost**, accesibles tanto desde una interfaz gráfica (Streamlit) como desde una línea de comandos (CLI).

## 🤖 Ejecucion Del Proyecto

En caso de estar en Windows y usar Visual Studio Code:
- Iniciar un ambiente virtual de python
- Instalar todos los requerimientos con este comando: pip install -r requirements.txt
- Iniciar el archivo de python app.py desde VSC

En caso de estar usando Linux/WSL:
- Iniciar un ambiente virtual de python
- Instalar todos los requerimientos con este comando: pip install -r requirements.txt
- Iniciar el archivo de python app.py con este comando: streamlit run app.py


---

## 📚 Revisión de la literatura (20%)

Diversos estudios recientes han abordado la relación entre las **propiedades acústicas** de una canción y su **popularidad en plataformas de streaming**.  
Trabajos como los de Ferraro et al. (2021) y Schedl et al. (2022) han mostrado que atributos como **energy**, **danceability**, **valence** y **tempo** presentan correlaciones significativas con métricas de popularidad.  
Otros enfoques emplean **redes neuronales** o **modelos híbridos** de análisis lírico y acústico (Ferwerda & Tkalčič, 2020), aunque estos requieren información textual o de contexto que no siempre está disponible.

Las brechas de investigación actuales se centran en:
- La **interpretabilidad** de los modelos de predicción de éxito musical.
- La **transferibilidad** de los resultados entre géneros y regiones.
- El uso de **datasets balanceados** y accesibles públicamente para reproducibilidad.

Este proyecto contribuye a esa línea mediante un **modelo interpretable** basado únicamente en *features acústicas*, con un dataset público de Spotify.

---

## 🎯 Pregunta de investigación y objetivos (15%)

**Pregunta principal:**  
> ¿Es posible predecir si una canción será popular en Spotify utilizando únicamente sus características acústicas disponibles mediante la API de audio features?

**Objetivo general:**  
Desarrollar y evaluar modelos de clasificación supervisada capaces de predecir la popularidad de una canción según sus atributos musicales cuantitativos.

**Objetivos específicos (SMART):**
1. **Recolectar y preparar** un dataset de canciones con variables numéricas y etiquetas de popularidad.
2. **Implementar y entrenar** modelos Random Forest y XGBoost, evaluando su rendimiento en términos de *accuracy*, *F1-score* y *ROC-AUC*.
3. **Identificar las características más influyentes** en la predicción de popularidad.
4. **Desplegar** una aplicación interactiva en Streamlit para facilitar el uso del modelo por parte de usuarios no técnicos.

---

## 📊 Datos y análisis preliminar (15%)

**Fuente de datos:**  
Dataset público de Kaggle: `spotify_sample_dataset.csv`, con 400 canciones provenientes de distintos artistas.

**Variables principales:**
| Tipo | Variables |
|------|------------|
| Identificación | name, artist, release_date |
| Características acústicas | danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo |
| Etiqueta | popularity (0–100) |

**Transformaciones realizadas:**
- Se generó una etiqueta binaria `label` donde:  
  - `1 = popular` si `popularity ≥ 50`  
  - `0 = no popular` en caso contrario.
- Las variables categóricas se codificaron con *one-hot encoding*.
- Se eliminaron columnas no predictivas (`name`, `artist`, `release_date`, `id`).

**Métrica de evaluación:**  
Se emplearon las métricas estándar de clasificación binaria:  
- **Accuracy** (exactitud global)  
- **F1-score** (balance entre precisión y exhaustividad)  
- **ROC-AUC** (área bajo la curva ROC, indicador de discriminación)

---

## ⚙️ Métodos (15%)

**Preprocesamiento:**
- Estandarización de variables mediante `StandardScaler`.
- División del dataset en conjuntos de entrenamiento y prueba con proporción 80/20 o configurable.
- Creación de *pipelines* reproducibles con `scikit-learn`.

**Modelos empleados:**
1. **Random Forest Classifier**  
   - `n_estimators=300`, `random_state=42`  
   - Permite evaluar la importancia de cada característica.
2. **XGBoost Classifier**  
   - `n_estimators=400`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.9`  
   - Optimización mediante boosting de gradiente.

**Validación:**
- Validación mediante *hold-out* (entrenamiento/prueba).  
- Las métricas se calcularon sobre el conjunto de prueba.  
- Para comparación, se consideró un **baseline teórico** equivalente a una clasificación aleatoria, que tendría un *accuracy* esperado ≈ 0.5.

---

## 📈 Resultados (15%)

Se entrenaron ambos modelos, pero el ejemplo reportado corresponde al **Random Forest** con un 50% de datos de prueba.

**Rendimiento obtenido:**
| Métrica | Valor |
|----------|--------|
| Accuracy | **0.840** |
| F1-score | **0.877** |
| ROC-AUC | **0.904** |

El siguiente ejemplo reportado corresponde al **XG_Boost** con un 50% de datos de prueba

**Rendimiento obtenido:**
| Métrica | Valor |
|----------|--------|
| Accuracy | **0.820** |
| F1-score | **0.856** |
| ROC-AUC | **0.912** |

Estos resultados superan ampliamente el baseline aleatorio, mostrando una buena capacidad de generalización.

**Características más relevantes según importancia del modelo:**
1. **danceability**
2. **energy**
3. **loudness**
4. **valence**
5. **tempo**

Las variables relacionadas con la “vivacidad” (*liveness*), el carácter instrumental (*instrumentalness*) y la “acousticness” tuvieron menor peso predictivo.

---

## 💬 Discusión (10%)

Los resultados indican que las características acústicas cuantitativas permiten capturar patrones que reflejan la **aceptación general del público** en Spotify.  
El modelo Random Forest alcanzó un rendimiento robusto (ROC-AUC 0.90), evidenciando una alta capacidad para distinguir entre canciones populares y no populares.

**Limitaciones:**
- El dataset es pequeño (400 canciones), lo cual puede limitar la generalización.
- No se consideraron variables contextuales como género musical, país o promoción.
- No se exploraron redes neuronales ni modelos de lenguaje para letras.

Aun así, el enfoque logra una **interpretabilidad clara**, permitiendo identificar las variables más influyentes sin requerir grandes recursos computacionales.

---

## 🧾 Conclusiones (10%)

- Se logró construir un modelo efectivo para predecir la popularidad de canciones basándose exclusivamente en sus características acústicas.  
- Las variables **tempo**, **valence**, **loudness**, **energy** y **danceability** resultaron ser los predictores más relevantes.  
- El modelo Random Forest superó ampliamente el baseline teórico, alcanzando una **precisión del 84%**.  
- La implementación dual (CLI + Streamlit) facilita tanto la experimentación técnica como el uso interactivo.

**Trabajo futuro:**
- Ampliar el dataset para incluir más géneros y periodos.
- Incorporar análisis lírico o semántico de letras.
- Experimentar con modelos híbridos (audio + texto).
- Desplegar el sistema en la nube para acceso público (por ejemplo, en AWS o HuggingFace Spaces).

---

## 🚀 Uso del proyecto sin app.py

**Modo consola:**
```bash
# Entrenamiento
python spotify_popularity_ml.py train --data spotify_sample_dataset.csv --algo random_forest --model_out modelo.pkl

# Predicción
python spotify_popularity_ml.py predict --model modelo.pkl --data nuevas_canciones.csv --out predicciones.csv
