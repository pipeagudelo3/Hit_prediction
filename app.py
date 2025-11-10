# -*- coding: utf-8 -*-
import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# XGBoost opcional
_XGB_OK = True
try:
    from xgboost import XGBClassifier
except Exception:
    _XGB_OK = False


# =========================
# Utilidades de datos
# =========================

NO_FEATURE_COLS = ['id', 'name', 'artist', 'release_date', 'popularity', 'label']

def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def one_hot_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Si existe 'genre' (u otra categórica simple), la convertimos a dummies."""
    work = df.copy()
    cat_cols = [c for c in ['genre'] if c in work.columns]
    if cat_cols:
        work = pd.get_dummies(work, columns=cat_cols, drop_first=True)
    return work

def preprocess_train(df_raw: pd.DataFrame):
    """
    Entrenamiento:
      - genera etiqueta binaria desde popularity (>=50 -> 1)
      - elimina columnas no útiles
      - aplica one-hot a 'genre' si existe
      - devuelve X (DataFrame numérico con columnas), y (np.array), feature_cols
    """
    df = df_raw.copy()
    if 'popularity' not in df.columns:
        raise ValueError("El CSV de entrenamiento debe incluir la columna 'popularity' (0-100).")
    df['label'] = (df['popularity'] >= 50).astype(int)

    # Drop columnas no features
    drop_cols = [c for c in NO_FEATURE_COLS if c in df.columns and c != 'popularity']
    Xdf = df.drop(drop_cols + ['popularity'], axis=1, errors='ignore')

    # One-hot si hay 'genre'
    Xdf = one_hot_if_needed(Xdf)

    # Quedarnos solo numéricas
    Xdf = Xdf.select_dtypes(include=[np.number])
    feature_cols = list(Xdf.columns)  # guardar orden/espacio de features

    X = Xdf.values
    y = df['label'].values
    return X, y, feature_cols

def preprocess_infer(df_raw: pd.DataFrame, feature_cols: list):
    """
    Inferencia:
      - elimina columnas no útiles
      - dummies si corresponde
      - reindexa a las columnas del entrenamiento (faltantes=0, extras descartadas)
      - devuelve X (np.array)
    """
    df = df_raw.copy()
    drop_cols = [c for c in NO_FEATURE_COLS if c in df.columns and c != 'popularity']
    Xdf = df.drop(drop_cols, axis=1, errors='ignore')

    Xdf = one_hot_if_needed(Xdf)
    Xdf = Xdf.select_dtypes(include=[np.number])

    # Alinear columnas con entrenamiento
    Xdf = Xdf.reindex(columns=feature_cols, fill_value=0)
    X = Xdf.values
    return X


# =========================
# Entrenamiento y métricas
# =========================

def build_model(algo: str):
    if algo == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
    elif algo == 'xgboost':
        if not _XGB_OK:
            raise RuntimeError("XGBoost no disponible. Instale con: pip install xgboost")
        clf = XGBClassifier(
            random_state=42,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric='logloss',
            tree_method='hist'
        )
    else:
        raise ValueError("Algoritmo no soportado")
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('classifier', clf)
    ])
    return pipe

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # predict_proba debería existir en RF/XGB
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    return acc, f1, roc


# =========================
# UI Streamlit
# =========================

st.set_page_config(page_title="Predicción de Popularidad Spotify", page_icon="🎵", layout="wide")
st.title("🎵 Predicción de Popularidad de Canciones (Spotify Audio Features)")

with st.sidebar:
    st.header("Modo de uso")
    modo = st.radio("Selecciona un modo:", ["Entrenar modelo", "Predecir con modelo"],
                    help="Puedes entrenar un modelo nuevo o cargar uno entrenado para predecir.")

st.markdown("""
**Tips de CSV**
- Entrenamiento: incluye `popularity` (0-100), y las columnas de audio (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, ...).  
- Predicción: **no** incluyas `popularity`.  
- Columnas como `name, artist, release_date` se ignoran automáticamente.
""")

if modo == "Entrenar modelo":
    st.subheader("1) Sube el CSV de entrenamiento (con `popularity`)")
    train_file = st.file_uploader("CSV de entrenamiento", type=['csv'], key="train_csv")

    col1, col2, col3 = st.columns(3)
    with col1:
        algo = st.selectbox("Algoritmo", ["random_forest", "xgboost"])
    with col2:
        test_size = st.slider("Proporción de test", 0.1, 0.4, 0.2, 0.05)
    with col3:
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

    if train_file is not None:
        df_train = load_csv(train_file)
        st.write("Vista previa del dataset:")
        st.dataframe(df_train.head(10), use_container_width=True)

        if st.button("🚀 Entrenar"):
            try:
                X, y, feature_cols = preprocess_train(df_train)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                model = build_model(algo)
                model.fit(X_train, y_train)
                acc, f1, roc = evaluate_model(model, X_test, y_test)

                st.success("Modelo entrenado correctamente.")
                st.write(f"**Accuracy:** {acc:.3f} | **F1:** {f1:.3f} | **ROC-AUC:** {roc:.3f}")

                # Importancias (si aplica)
                try:
                    clf = model.named_steps['classifier']
                    if hasattr(clf, "feature_importances_"):
                        importances = clf.feature_importances_
                        imp_df = pd.DataFrame({
                            "feature": feature_cols,
                            "importance": importances
                        }).sort_values("importance", ascending=False).head(20)
                        st.markdown("**Top 20 features más importantes:**")
                        st.dataframe(imp_df, use_container_width=True)
                except Exception:
                    pass

                # Empaquetar para descargar: guardamos pipeline + feature_cols
                payload = {"pipeline": model, "feature_cols": feature_cols}
                buffer = io.BytesIO()
                joblib.dump(payload, buffer)
                st.download_button(
                    label="💾 Descargar modelo entrenado (.pkl)",
                    data=buffer.getvalue(),
                    file_name=f"modelo_spotify_{algo}.pkl",
                    mime="application/octet-stream"
                )

            except Exception as e:
                st.error(f"Error durante entrenamiento: {e}")

elif modo == "Predecir con modelo":
    st.subheader("1) Carga el modelo entrenado (.pkl)")
    model_file = st.file_uploader("Modelo .pkl (descargado de la sección de entrenamiento)", type=['pkl'], key="model_pkl")

    st.subheader("2) Sube el CSV de canciones nuevas (sin `popularity`)")
    infer_file = st.file_uploader("CSV de nuevas canciones", type=['csv'], key="infer_csv")

    if model_file is not None:
        try:
            payload = joblib.load(model_file)
            model: Pipeline = payload["pipeline"]
            feature_cols: list = payload["feature_cols"]
            st.success("Modelo cargado.")
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
            st.stop()

        if infer_file is not None:
            df_new = load_csv(infer_file)
            st.write("Vista previa de las canciones a predecir:")
            st.dataframe(df_new.head(10), use_container_width=True)

            if st.button("🔮 Predecir popularidad"):
                try:
                    X_new = preprocess_infer(df_new, feature_cols)
                    preds = model.predict(X_new)
                    proba = model.predict_proba(X_new)[:, 1]

                    out = df_new.copy()
                    out['pred_popular'] = preds
                    out['pred_proba'] = np.round(proba, 4)

                    st.success("Predicciones listas.")
                    st.dataframe(out.head(20), use_container_width=True)

                    # Descargar CSV con predicciones
                    csv_bytes = out.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Descargar predicciones CSV",
                        data=csv_bytes,
                        file_name="predicciones_spotify.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error durante la predicción: {e}")
