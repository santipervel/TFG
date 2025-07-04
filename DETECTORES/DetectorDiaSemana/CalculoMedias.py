import pandas as pd
from pathlib import Path
import numpy as np

# Cargar CSV de resultados completos (con días de semana)
ruta_script = Path(__file__).resolve().parent
ruta_csv = ruta_script.parent / "DetectorDiaSemana" / "Resultados_Inyeccion_2D" / "Resultados_100iteraciones_TodosUsuarios_2D.csv"  # cambia si es otro nombre
df = pd.read_csv(ruta_csv, sep=";")

# --- Calcular medias globales (sin iteraciones ni usuarios) ---
df_media = df.groupby(["modelo", "normalizado_manual"]).agg(
    TP=("TP", "mean"),
    FN=("FN", "mean"),
    FP=("FP", "mean"),
    TN=("TN", "mean"),
    TPR=("TPR", "mean"),
    #Rendimiento=("Rendimiento", "mean")
).reset_index()

# --- Calcular FPR y TNR ---
df_media["FPR"] = df_media["FP"] / (df_media["FP"] + df_media["TN"])
df_media["TNR"] = df_media["TN"] / (df_media["FP"] + df_media["TN"])
df_media["Rendimiento"] = np.sqrt(df_media["TPR"] * df_media["TNR"])

# --- Mostrar resultados ---
print("\n✅ Tabla de métricas medias (todos usuarios, todas iteraciones, todos días)")
print(df_media)
