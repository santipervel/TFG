import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns

# Cargar resultados
ruta_script = Path(__file__).resolve().parent
ruta_csv = ruta_script.parent / "DetectorDiaSemana" / "Resultados_Inyeccion_2D" / "metricas_KNN_normalizado_todosUsus.csv"  # cambia si es otro nombre
df = pd.read_csv(ruta_csv, sep=";")

# Filtrar para svm sin normalizar
#df_svm = df[(df["modelo"] == "knn") & (df["normalizado_manual"] == False)]

# Calcular rendimiento medio por usuario
df_ranking = df.groupby("usuario").agg(
    TP=("TP", "mean"),
    FN=("FN", "mean"),
    FP=("FP", "mean"),
    TN=("TN", "mean"),
    TPR=("TPR", "mean"),
).reset_index()
df_ranking["FPR"] = df_ranking["FP"] / (df_ranking["FP"] + df_ranking["TN"])
df_ranking["TNR"] = df_ranking["TN"] / (df_ranking["FP"] + df_ranking["TN"])
df_ranking["Rendimiento"] = np.sqrt(df_ranking["TPR"] * df_ranking["TNR"])

# Ordenar y seleccionar top 10 mejores y 10 peores
df_sorted = df_ranking.sort_values(by="Rendimiento", ascending=False)
top_10_mejores = df_sorted.head(10)["usuario"].tolist()
top_10_peores = df_sorted.tail(10)["usuario"].tolist()

# Mostrar resultados
print("\nTop 10 mejores usuarios (svm sin normalizar):")
print(top_10_mejores)
print("\nTop 10 peores usuarios (svm sin normalizar):")
print(top_10_peores)
