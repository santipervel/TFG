import pandas as pd
from pathlib import Path
import numpy as np
from itertools import product

# --- CONFIGURACIÓN ---
normalizar = True
valores_k = [0.5, 1.0, 1.5, 2.0]  # Valores de k para la detección  1.0, 1.5, 2.0
factores = [5.0, 8.0, 10.0, 12.0 ]   # Factores de amplificación para la inyección de anomalías

#Combinación óptima
# valores_k = [0.5]
# factores = [12]

num_anomalias = 200
num_iteraciones = 10

# --- Rutas ---
ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]
modo = "normalizado" if normalizar else "original"
output_dir = ruta_script / "Perfil_Horario_EventosUsuarios" / modo
output_dir.mkdir(parents=True, exist_ok=True)
ruta_usuarios = ruta_script.parent / "DetectorPerfilComportamiento" / "usuarios_seleccionados" / "top_100_usuarios_2018.csv"

eventos_interes = [
    "file_accessed", "file_written", "file_updated", "file_created",
    "login_attempt", "login_successful", "version_deleted",
    "public_share_password_changed", "deleted_from_trashbin",
    "file_deleted", "file_renamed", "public_share_accessed"
]

usuarios_total = pd.read_csv(ruta_usuarios, sep=";")["uid"].tolist()

# --- Cargar datos ---
df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])
df["time"] = pd.to_datetime(df["time"])
df = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")]
df["hora"] = df["time"].dt.hour
df["fecha"] = df["time"].dt.date

# --- Filtrar entrenamiento ---
df_train = df[
    (df["time"] < "2018-07-01") &
    (df["uid"].isin(usuarios_total)) &
    (df["type"].isin(eventos_interes))
].copy()

fechas = pd.date_range("2018-01-01", "2018-06-30", freq="D")
horas = list(range(24))
calendario = pd.DataFrame(list(product(fechas.date, horas)), columns=["fecha", "hora"])

# --- Construcción del perfil horario ---
registros = []
for usuario in usuarios_total:
    for evento in eventos_interes:
        df_sub = df_train[(df_train["uid"] == usuario) & (df_train["type"] == evento)]
        if df_sub.empty:
            continue
        conteo = df_sub.groupby(["fecha", "hora"]).size().reset_index(name="eventos")
        base = calendario.copy()
        base["uid"] = usuario
        base["type"] = evento
        base = base.merge(conteo, on=["fecha", "hora"], how="left")
        base["eventos"] = base["eventos"].fillna(0).astype(int)
        registros.append(base)

df_perfiles = pd.concat(registros, ignore_index=True)
perfil_final = (
    df_perfiles.groupby(["uid", "type", "hora"])["eventos"]
    .agg(["mean", "std"]).reset_index()
    .rename(columns={"mean": "media_eventos", "std": "std_eventos"})
)
perfil_final["std_eventos"] = perfil_final["std_eventos"].fillna(0).round(2)
perfil_final["media_eventos"] = perfil_final["media_eventos"].round(2)
perfil_final.to_csv(output_dir / "perfil_horario_usuarios_eventos.csv", index=False, sep=";") 

# --- FASE DE DETECCIÓN MULTI-ITERACIÓN ---
fechas_test = pd.date_range("2018-07-01", "2018-12-31", freq="D")
calendario_test = pd.DataFrame(list(product(fechas_test.date, horas)), columns=["fecha", "hora"])

df_test = df[
    (df["time"] >= "2018-07-01") &
    (df["uid"].isin(usuarios_total)) &
    (df["type"].isin(eventos_interes))
].copy()

for k in valores_k:
    for factor in factores:
        print(f"⏱ Ejecutando {num_iteraciones} iteraciones para k={k}, factor={factor}")
        todas_iteraciones = []

        for i in range(num_iteraciones):
            print(f"Iteración {i + 1}/{num_iteraciones}")
            metricas = []
            for usuario in usuarios_total:
                for evento in eventos_interes:
                    df_sub_test = df_test[(df_test["uid"] == usuario) & (df_test["type"] == evento)]
                    if df_sub_test.empty:
                        continue
                    conteo_test = df_sub_test.groupby(["fecha", "hora"]).size().reset_index(name="eventos")
                    base_test = calendario_test.copy()
                    base_test["uid"] = usuario
                    base_test["type"] = evento
                    base_test = base_test.merge(conteo_test, on=["fecha", "hora"], how="left")
                    base_test["eventos"] = base_test["eventos"].fillna(0).astype(int)
                    base_test["label"] = 0

                    if len(base_test) < num_anomalias:
                        continue

                    seleccionadas = base_test.sample(n=num_anomalias, random_state=np.random.randint(100000)).copy()
                    indices_modificar = seleccionadas.index

                    perfil_usuario_evento = perfil_final[
                        (perfil_final["uid"] == usuario) & (perfil_final["type"] == evento)
                    ]
                    if perfil_usuario_evento.empty:
                        continue

                    seleccionadas = seleccionadas.merge(perfil_usuario_evento, on=["uid", "type", "hora"], how="left")
                    seleccionadas.loc[seleccionadas["media_eventos"] == 0, "media_eventos"] = 1
                    seleccionadas["eventos"] = (seleccionadas["eventos"] + factor * seleccionadas["media_eventos"]).round().astype(int)
                    seleccionadas["label"] = 1

                    base_test.loc[indices_modificar, "eventos"] = seleccionadas["eventos"].values
                    base_test.loc[indices_modificar, "label"] = 1

                    base_test = base_test.merge(perfil_usuario_evento, on=["uid", "type", "hora"], how="left")
                    base_test["media_eventos"] = base_test["media_eventos"].fillna(0)
                    base_test["std_eventos"] = base_test["std_eventos"].fillna(0)

                    if normalizar:
                        base_test["z"] = np.nan
                        mask_std = base_test["std_eventos"] > 0
                        base_test.loc[mask_std, "z"] = (
                            (base_test.loc[mask_std, "eventos"] - base_test.loc[mask_std, "media_eventos"]) /
                            base_test.loc[mask_std, "std_eventos"]
                        )
                        mask_zero_std = (base_test["std_eventos"] == 0) & (base_test["eventos"] > base_test["media_eventos"])
                        base_test.loc[mask_zero_std, "z"] = 999
                        base_test["z"] = base_test["z"].replace([np.inf, -np.inf], 0).fillna(0)
                        base_test["es_anomalia"] = base_test["z"] > k
                    else:
                        base_test["umbral"] = base_test["media_eventos"] + k * base_test["std_eventos"]
                        base_test["es_anomalia"] = base_test["eventos"] > base_test["umbral"]

                    TP = ((base_test["label"] == 1) & (base_test["es_anomalia"] == True)).sum()
                    FN = ((base_test["label"] == 1) & (base_test["es_anomalia"] == False)).sum()
                    FP = ((base_test["label"] == 0) & (base_test["es_anomalia"] == True)).sum()
                    TN = ((base_test["label"] == 0) & (base_test["es_anomalia"] == False)).sum()

                    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                    TNR = TN / (FP + TN) if (FP + TN) > 0 else 0
                    rendimiento = np.sqrt(TPR * TNR)

                    metricas.append({
                        "usuario": usuario, "evento": evento, "TP": TP, "FN": FN, "FP": FP, "TN": TN,
                        "TPR": TPR, "FPR": FPR, "TNR": TNR, "Rendimiento": rendimiento
                    })
            df_metricas = pd.DataFrame(metricas)
            todas_iteraciones.append(df_metricas)

        # --- Guardar métricas medias por evento y global ---
        df_concat = pd.concat(todas_iteraciones)
        df_evento = df_concat.groupby("evento").mean(numeric_only=True).reset_index()
        df_evento[["TP", "FN", "FP", "TN"]] = df_evento[["TP", "FN", "FP", "TN"]].round(0).astype(int)
        df_evento[["TPR", "FPR", "TNR", "Rendimiento"]] = df_evento[["TPR", "FPR", "TNR", "Rendimiento"]].round(4)
        df_evento.to_csv(output_dir / f"metricas_medias_por_evento_k_{k}_factor_{factor}.csv", sep=";", index=False)

        # --- Guardar métricas globales (una fila por combinación) ---
        df_media = df_concat.mean(numeric_only=True).to_frame().T
        df_media[["TP", "FN", "FP", "TN"]] = df_media[["TP", "FN", "FP", "TN"]].round(0).astype(int)
        df_media[["TPR", "FPR", "TNR", "Rendimiento"]] = df_media[["TPR", "FPR", "TNR", "Rendimiento"]].round(4)
        df_media["k"] = k
        df_media["factor_anomalia"] = factor

        ruta_salida = output_dir / f"resultados_medias_globales_{modo}.csv"
        df_media.to_csv(ruta_salida, sep=";", index=False, mode="a", header=not ruta_salida.exists())


print("✅ Evaluación completada.")
