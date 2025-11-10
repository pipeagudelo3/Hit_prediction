# -*- coding: utf-8 -*-
"""
Sistema base para predecir si una canción será popular usando features de audio de Spotify.
- Entrena y guarda un modelo (RandomForest o XGBoost)
- Carga un modelo guardado y predice sobre canciones nuevas
- Muestra métricas (Accuracy, F1, ROC-AUC)

Uso desde consola:
    python spotify_popularity_ml.py
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# XGBoost es opcional. Si no está instalado, se ignora esa opción.
_XGB_AVAILABLE = True
try:
    from xgboost import XGBClassifier
except Exception:
    _XGB_AVAILABLE = False


# ---------- CARGA Y PREPROCESAMIENTO ----------

def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga un CSV en un DataFrame de pandas."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    return pd.read_csv(ruta)


def preprocesar_datos(df: pd.DataFrame, entrenamiento: bool = True):
    """
    Preprocesa los datos:
    - En entrenamiento: crea etiqueta binaria desde 'popularity' (>=50 => 1), retorna X, y
    - En inferencia: elimina columnas irrelevantes y retorna X
    - Si hubiera categóricas tipo 'genre', se pueden one-hot-encode aquí
    """
    df_proc = df.copy()

    if entrenamiento:
        if 'popularity' not in df_proc.columns:
            raise ValueError("En entrenamiento, el CSV debe tener columna 'popularity' (0-100).")
        df_proc['label'] = (df_proc['popularity'] >= 50).astype(int)
        y = df_proc['label'].values
        df_proc = df_proc.drop(['popularity', 'label'], axis=1)
    else:
        y = None
        if 'popularity' in df_proc.columns:
            df_proc = df_proc.drop(['popularity'], axis=1)

    # Eliminar columnas no numéricas/irrelevantes comunes
    cols_drop = [c for c in ['id', 'name', 'artist', 'release_date'] if c in df_proc.columns]
    if cols_drop:
        df_proc = df_proc.drop(cols_drop, axis=1)

    # One-hot para categóricas opcionales (p.ej., 'genre')
    if 'genre' in df_proc.columns:
        df_proc = pd.get_dummies(df_proc, columns=['genre'], drop_first=True)

    # Asegurar sólo columnas numéricas
    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns
    X = df_proc[numeric_cols].values
    return (X, y) if entrenamiento else X


# ---------- ENTRENAMIENTO Y EVALUACIÓN ----------

def entrenar_modelo(X_train: np.ndarray, y_train: np.ndarray, modelo: str = 'random_forest') -> Pipeline:
    """
    Entrena un pipeline con StandardScaler + clasificador.
    modelo: 'random_forest' o 'xgboost'
    """
    if modelo not in ('random_forest', 'xgboost'):
        raise ValueError("Modelo no soportado. Use 'random_forest' o 'xgboost'.")

    if modelo == 'xgboost' and not _XGB_AVAILABLE:
        raise RuntimeError("XGBoost no está instalado. Instale con: pip install xgboost")

    if modelo == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    else:
        clf = XGBClassifier(
            random_state=42,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric='logloss',
            tree_method='hist'  # rápido en CPU
        )

    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('classifier', clf)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluar_modelo(modelo: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Calcula e imprime Accuracy, F1 y ROC-AUC. Devuelve dict con los resultados."""
    y_pred = modelo.predict(X_test)
    # Algunos clasificadores pueden no tener predict_proba (estos sí lo tienen)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"\nMétricas en test:")
    print(f" - Accuracy : {acc:.3f}")
    print(f" - F1-score : {f1:.3f}")
    print(f" - ROC-AUC  : {roc:.3f}")

    return {'accuracy': acc, 'f1': f1, 'roc_auc': roc}


# ---------- GUARDADO / CARGA ----------

def guardar_modelo(modelo: Pipeline, ruta: str):
    """Guarda el pipeline en un archivo .pkl."""
    joblib.dump(modelo, ruta)
    print(f"\n✅ Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str) -> Pipeline:
    """Carga un pipeline .pkl desde disco."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {ruta}")
    modelo = joblib.load(ruta)
    print(f"✅ Modelo cargado desde: {ruta}")
    return modelo


# ---------- PREDICCIÓN ----------

def predecir(modelo: Pipeline, df_nuevos: pd.DataFrame) -> np.ndarray:
    """
    Predice (0/1) para nuevas canciones. df_nuevos NO debe traer 'popularity'.
    Devuelve array de 0=No popular, 1=Popular.
    """
    X_new = preprocesar_datos(df_nuevos, entrenamiento=False)
    preds = modelo.predict(X_new)
    return preds


# ---------- CLI / MAIN ----------

def cli():
    parser = argparse.ArgumentParser(description="Predicción de popularidad de canciones (Spotify features)")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcomando: entrenar
    p_train = subparsers.add_parser('train', help='Entrenar y guardar un modelo')
    p_train.add_argument('--data', required=True, help='Ruta CSV de entrenamiento (con columna popularity)')
    p_train.add_argument('--model_out', default='modelo_spotify.pkl', help='Ruta de salida del modelo .pkl')
    p_train.add_argument('--algo', choices=['random_forest', 'xgboost'], default='random_forest',
                         help="Algoritmo a usar (default: random_forest)")
    p_train.add_argument('--test_size', type=float, default=0.2, help='Proporción test (default 0.2)')

    # Subcomando: predecir
    p_pred = subparsers.add_parser('predict', help='Cargar modelo y predecir nuevas canciones')
    p_pred.add_argument('--model', required=True, help='Ruta del modelo .pkl')
    p_pred.add_argument('--data', required=True, help='Ruta CSV con nuevas canciones (sin popularity)')
    p_pred.add_argument('--out', help='(Opcional) Ruta CSV para guardar predicciones')

    args = parser.parse_args()

    if args.command == 'train':
        print("🚀 Entrenando modelo...")
        df = cargar_datos(args.data)
        X, y = preprocesar_datos(df, entrenamiento=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        modelo = entrenar_modelo(X_train, y_train, modelo=args.algo)
        _ = evaluar_modelo(modelo, X_test, y_test)
        guardar_modelo(modelo, args.model_out)

    elif args.command == 'predict':
        print("🔮 Prediciendo sobre nuevas canciones...")
        modelo = cargar_modelo(args.model)
        df_new = cargar_datos(args.data)
        preds = predecir(modelo, df_new)
        # Mostrar por consola
        print("\nPredicciones (0 = No popular, 1 = Popular):")
        print(preds)

        # Guardar si se pidió
        if args.out:
            out_df = df_new.copy()
            out_df['pred_popular'] = preds
            out_df.to_csv(args.out, index=False)
            print(f"\n✅ Predicciones guardadas en: {args.out}")


if __name__ == "__main__":
    # Si se ejecuta sin argumentos, mostramos ayuda amigable
    if len(sys.argv) == 1:
        print("""
=== Predicción de Popularidad de Canciones (Spotify) ===

Ejemplos de uso:

1) Entrenar un modelo RandomForest y guardarlo:
   python spotify_popularity_ml.py train --data spotify_sample_dataset.csv --algo random_forest --model_out modelo_spotify.pkl

   (Para XGBoost, primero instale: pip install xgboost)
   python spotify_popularity_ml.py train --data spotify_sample_dataset.csv --algo xgboost --model_out modelo_spotify_xgb.pkl

2) Predecir con un modelo guardado sobre canciones nuevas (sin 'popularity'):
   python spotify_popularity_ml.py predict --model modelo_spotify.pkl --data nuevas_canciones.csv --out predicciones.csv
""")
    else:
        cli()
