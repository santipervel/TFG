import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import random

# --- Configuración ---
window_sizes = [5, 10, 20, 30, 40, 50]
alphas = [0.3, 0.6, 0.9]
ks = [0, 0.1, 0.3, 0.5]             # Para detección
deltas = [1, 2, 3]             # Para inserción de anomalías

#Combinación óptima
# window_sizes = [5]
# alphas = [0.3]
# ks =  [0.3]             
# deltas = [3]     
        
simulaciones = 30
anomalias_por_simulacion = 100

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
eventos_interes = [
    "file_accessed", "file_written", "file_updated", "file_created",
    "login_attempt", "login_successful", "version_deleted",
    "public_share_password_changed", "deleted_from_trashbin",
    "file_deleted", "file_renamed", "public_share_accessed"
]

# --- Salida ---
output_dir = ruta_script / "Resultados_EWMA_tipo_evento"
output_dir.mkdir(parents=True, exist_ok=True)

resultados_globales = []
resultados_por_usuario = []

for window_size, alpha, k, delta in product(window_sizes, alphas, ks, deltas):
    print(f"🔁 Ejecutando win={window_size}, alpha={alpha}, k={k}, delta={delta}")
    total_TP, total_FN, total_FP, total_TN = [], [], [], []

    for usuario in usuarios_total:
        df_u = df[df["uid"] == usuario]
        if df_u.empty:
            continue

        for evento in eventos_interes:
            df_sub = df_u[df_u["type"] == evento]
            if df_sub.empty:
                continue

            df_sub = df_sub.sort_values("time").set_index("time")
            df_agg_base = df_sub.resample(f"{window_size}min").size().reset_index(name="eventos")

            if len(df_agg_base) < 110:
                continue

            media_eventos = df_agg_base["eventos"].mean()
            std_eventos = df_agg_base["eventos"].std()

            TP_e, FN_e, FP_e, TN_e = [], [], [], []

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

                TP_e.append(TP)
                FN_e.append(FN)
                FP_e.append(FP)
                TN_e.append(TN)

            if TP_e:
                avg_TP = np.mean(TP_e)
                avg_FN = np.mean(FN_e)
                avg_FP = np.mean(FP_e)
                avg_TN = np.mean(TN_e)
                TPR = round(avg_TP / (avg_TP + avg_FN), 4) if (avg_TP + avg_FN) > 0 else 0
                FPR = round(avg_FP / (avg_FP + avg_TN), 4) if (avg_FP + avg_TN) > 0 else 0
                TNR = round(avg_TN / (avg_FP + avg_TN), 4) if (avg_FP + avg_TN) > 0 else 0
                rendimiento = np.sqrt(TPR * TNR)

                resultados_por_usuario.append({
                    "usuario": usuario,
                    "evento": evento,
                    "window_size": window_size,
                    "alpha": alpha,
                    "k": k,
                    "delta": delta,
                    "TPR": TPR,
                    "FPR": FPR,
                    "TNR": TNR,
                    "rendimiento": round(rendimiento, 4)
                })

                total_TP.append(avg_TP)
                total_FN.append(avg_FN)
                total_FP.append(avg_FP)
                total_TN.append(avg_TN)

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
# --- Calcular métricas medias por tipo de evento ---
df_usuario_evento = pd.DataFrame(resultados_por_usuario)
resumen_por_evento = (
    df_usuario_evento
    .groupby(["evento", "window_size", "alpha", "k", "delta"])
    .agg({
        "TPR": "mean",
        "FPR": "mean",
        "TNR": "mean",
        "rendimiento": "mean"
    })
    .reset_index()
    .sort_values(["evento", "window_size", "alpha", "k", "delta"])
)

resumen_por_evento.to_csv(output_dir / "resumen_metricas_por_tipo.csv", sep=";", index=False)

# --- Guardar CSVs ---
pd.DataFrame(resultados_globales).to_csv(output_dir / "resumen_global_metricas_por_tipo.csv", sep=";", index=False)
pd.DataFrame(resultados_por_usuario).to_csv(output_dir / "resumen_metricas_por_usuario_y_tipo.csv", sep=";", index=False)

print("✅ CSVs de resumen global y por usuario y tipo generados correctamente.")
