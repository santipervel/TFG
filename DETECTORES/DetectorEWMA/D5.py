import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import random

# --- Configuración inicial ---
window_sizes = [5, 10, 20, 30, 40, 50]
alphas = [0.3, 0.6, 0.9]
ks = [0, 0.1, 0.3, 0.5]               
deltas = [1, 2, 3]           

#Combinación óptima 
# window_sizes = [5]  
# alphas = [0.6]
# ks = [0.5]               
# deltas = [3] 

simulaciones = 30
anomalias_por_simulacion = 100

# --- Rutas ---
ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]
ruta_usuarios = ruta_script.parent / "DetectorPerfilComportamiento" / "usuarios_seleccionados" / "top_100_usuarios_2018.csv"

# --- Cargar datos ---
print("📥 Cargando datos...")
df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])
df["time"] = pd.to_datetime(df["time"])
df = df[(df["time"] >= "2018-07-01") & (df["time"] < "2019-01-01")]

usuarios_total = pd.read_csv(ruta_usuarios, sep=";")["uid"].tolist()

# --- Crear carpeta de salida ---
output_dir = ruta_script / "Resultados_EWMA"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Almacén de resultados ---
resultados_globales = []
resultados_por_usuario = []

# --- Bucle por parámetros ---
for window_size, alpha, k, delta in product(window_sizes, alphas, ks, deltas):
    print(f"🔁 Ejecutando para win={window_size}, alpha={alpha}, k={k}, delta={delta}")

    total_TP, total_FN, total_FP, total_TN = [], [], [], []

    for usuario in usuarios_total:
        df_usuario = df[df["uid"] == usuario].copy()
        df_usuario = df_usuario.sort_values("time").set_index("time")
        df_agg_base = df_usuario.resample(f"{window_size}min").size().reset_index(name="eventos")

        if len(df_agg_base) < 110:
            continue

        media_eventos = df_agg_base["eventos"].mean()
        std_eventos = df_agg_base["eventos"].std()

        TP_user, FN_user, FP_user, TN_user = [], [], [], []

        for _ in range(simulaciones):
            df_agg = df_agg_base.copy()
            total_indices = df_agg.index.tolist()
            if len(total_indices) < anomalias_por_simulacion:
                continue

            seleccion = random.sample(total_indices, k=anomalias_por_simulacion)
            df_agg["eventos_mod"] = df_agg["eventos"].astype(float)

            for idx in seleccion:
                df_agg.loc[idx, "eventos_mod"] +=  delta * std_eventos

            eventos_shift = df_agg["eventos_mod"].shift(1)
            df_agg["ewma"] = eventos_shift.ewm(alpha=alpha, adjust=False).mean()
            df_agg["std"] = eventos_shift.expanding().std()
            df_agg["umbral_sup"] = df_agg["ewma"] + k * df_agg["std"]
            df_agg["anomalía"] = df_agg["eventos_mod"] > df_agg["umbral_sup"]

            etiquetas = np.zeros(len(df_agg), dtype=int)
            for idx in seleccion:
                etiquetas[idx] = 1

            predicciones = df_agg["anomalía"].astype(int).values
            TP = np.sum((etiquetas == 1) & (predicciones == 1))
            FN = np.sum((etiquetas == 1) & (predicciones == 0))
            FP = np.sum((etiquetas == 0) & (predicciones == 1))
            TN = np.sum((etiquetas == 0) & (predicciones == 0))

            TP_user.append(TP)
            FN_user.append(FN)
            FP_user.append(FP)
            TN_user.append(TN)

        if TP_user:
            avg_TP_u = np.mean(TP_user)
            avg_FN_u = np.mean(FN_user)
            avg_FP_u = np.mean(FP_user)
            avg_TN_u = np.mean(TN_user)
            TPR_u = round(avg_TP_u / (avg_TP_u + avg_FN_u), 4) if (avg_TP_u + avg_FN_u) > 0 else 0
            FPR_u = round(avg_FP_u / (avg_FP_u + avg_TN_u), 4) if (avg_FP_u + avg_TN_u) > 0 else 0
            TNR_u = round(avg_TN_u / (avg_FP_u + avg_TN_u), 4) if (avg_FP_u + avg_TN_u) > 0 else 0
            rendimiento_u = np.sqrt(TPR_u * TNR_u)

            resultados_por_usuario.append({
                "usuario": usuario,
                "window_size": window_size,
                "alpha": alpha,
                "k": k,
                "delta": delta,
                "TPR": TPR_u,
                "FPR": FPR_u,
                "TNR": TNR_u,
                "rendimiento": round(rendimiento_u, 4)
            })

            total_TP.append(avg_TP_u)
            total_FN.append(avg_FN_u)
            total_FP.append(avg_FP_u)
            total_TN.append(avg_TN_u)

    # --- Métricas globales ---
    if total_TP:
        avg_TP = np.mean(total_TP)
        avg_FN = np.mean(total_FN)
        avg_FP = np.mean(total_FP)
        avg_TN = np.mean(total_TN)
        TPR = round(avg_TP / (avg_TP + avg_FN), 4) if (avg_TP + avg_FN) > 0 else 0
        FPR = round(avg_FP / (avg_FP + avg_TN), 4) if (avg_FP + avg_TN) > 0 else 0
        TNR = round(avg_TN / (avg_FP + avg_TN), 4) if (avg_FP + avg_TN) > 0 else 0
        rendimiento = np.sqrt(TPR * TNR)

        resultados_globales.append({
            "window_size": window_size,
            "alpha": alpha,
            "k": k,
            "delta": delta,
            "avg_TP": round(avg_TP, 2),
            "avg_FN": round(avg_FN, 2),
            "avg_FP": round(avg_FP, 2),
            "avg_TN": round(avg_TN, 2),
            "TPR": TPR,
            "TNR": TNR,
            "FPR": FPR,
            "rendimiento": round(rendimiento, 4)
        })

# --- Guardar CSVs ---
df_global = pd.DataFrame(resultados_globales)
df_global.sort_values(["window_size", "alpha", "k", "delta"], inplace=True)
df_global.to_csv(output_dir / "resumen_global_metricas.csv", sep=";", index=False)

df_usuario = pd.DataFrame(resultados_por_usuario)
df_usuario.sort_values(["window_size", "alpha", "k", "delta", "usuario"], inplace=True)
df_usuario.to_csv(output_dir / "resumen_metricas_por_usuario.csv", sep=";", index=False)

print("✅ CSVs de resumen global y por usuario generados correctamente.")
