import pandas as pd
import numpy as np
from pathlib import Path
import random

# --- Parámetros generales ---
ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]
ruta_usuarios = ruta_script / "usuarios_seleccionados" / "top_100_usuarios_2018.csv"

num_anomalias = 100
NUM_ITERACIONES = 30
normalizar = False
valores_k = [0.25, 0.5, 1.0, 1.5, 2.0]

# --- Cargar datos base ---
print("📥 Cargando dataset completo...")
df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])
df["time"] = pd.to_datetime(df["time"])
df = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")]

usuarios_lista = pd.read_csv(ruta_usuarios, sep=";")["uid"].tolist()
resultados_globales = []

for usuario in usuarios_lista:
    print(f"\n👤 Procesando usuario: {usuario}")
    df_user = df[df["uid"] == usuario].copy()
    if df_user.empty:
        continue

    df_user["fecha"] = df_user["time"].dt.date
    df_user["hora"] = df_user["time"].dt.hour
    df_user["dia_semana"] = df_user["time"].dt.day_name()

    df_train = df_user[df_user["time"] < "2018-07-01"].copy()
    df_test_real = df_user[df_user["time"] >= "2018-07-01"].copy()
    if df_train.empty or df_test_real.empty:
        continue

    fechas_entrenamiento = pd.date_range("2018-01-01", "2018-06-30", freq="D")
    fechas_test = pd.date_range("2018-07-01", "2018-12-31", freq="D")
    horas = list(range(24))

    df_base_train = pd.DataFrame([(d.date(), h) for d in fechas_entrenamiento for h in horas], columns=["fecha", "hora"])
    df_base_train["dia_semana"] = pd.to_datetime(df_base_train["fecha"]).dt.day_name()

    eventos_train = df_train.groupby(["fecha", "hora"]).size().reset_index(name="eventos")
    df_completo_train = pd.merge(df_base_train, eventos_train, how="left", on=["fecha", "hora"])
    df_completo_train["eventos"] = df_completo_train["eventos"].fillna(0).astype(int)

    perfil = df_completo_train.groupby(["dia_semana", "hora"]).agg(
        media_eventos=("eventos", "mean"),
        std_eventos=("eventos", "std")
    ).reset_index()
    perfil["std_eventos"] = perfil["std_eventos"].fillna(0)

    df_base_test = pd.DataFrame([(d.date(), h) for d in fechas_test for h in horas], columns=["fecha", "hora"])
    df_base_test["dia_semana"] = pd.to_datetime(df_base_test["fecha"]).dt.day_name()

    eventos_test = df_test_real.groupby(["fecha", "hora"]).size().reset_index(name="eventos")
    df_completo_test = pd.merge(df_base_test, eventos_test, how="left", on=["fecha", "hora"])
    df_completo_test["eventos"] = df_completo_test["eventos"].fillna(0).astype(int)

    resultados = []

    for k in valores_k:
        metricas_iter = []

        for _ in range(NUM_ITERACIONES):
            df_inyectado = df_completo_test.copy()
            df_inyectado["label"] = 0

            candidatas = df_inyectado[df_inyectado["eventos"] == 0].copy()
            candidatas = candidatas.sample(n=min(num_anomalias, len(candidatas)), random_state=np.random.randint(100000))

            for idx, fila in candidatas.iterrows():
                fecha_actual = pd.to_datetime(fila["fecha"])
                hora_actual = fila["hora"]

                fecha_actual = pd.to_datetime(fila["fecha"])
                hora_actual = fila["hora"]

                # Buscar en el mismo día, horas anteriores con eventos > 0
                posibles = df_completo_test[
                    (pd.to_datetime(df_completo_test["fecha"]) == fecha_actual) &
                    (df_completo_test["hora"] < hora_actual) &
                    (df_completo_test["eventos"] > 0)
                ].sort_values(by="hora", ascending=False)

                if not posibles.empty:
                    eventos_copiados = posibles.iloc[0]["eventos"]
                    df_inyectado.at[idx, "eventos"] = eventos_copiados
                    df_inyectado.at[idx, "label"] = 1


            anomalias_inyectadas = df_inyectado["label"].sum()

            df_eval = pd.merge(df_inyectado, perfil, how="left", on=["dia_semana", "hora"])

            if normalizar:
                df_eval["eventos_normalizados"] = (df_eval["eventos"] - df_eval["media_eventos"]) / df_eval["std_eventos"]
                df_eval["eventos_normalizados"] = df_eval["eventos_normalizados"].fillna(0)
                df_eval["es_anomalia"] = df_eval["eventos_normalizados"] > k
            else:
                df_eval["umbral"] = df_eval["media_eventos"] + k * df_eval["std_eventos"]
                df_eval["es_anomalia"] = df_eval["eventos"] > df_eval["umbral"]

            TP = ((df_eval["label"] == 1) & (df_eval["es_anomalia"] == True)).sum()
            FN = anomalias_inyectadas - TP
            FP = ((df_eval["label"] == 0) & (df_eval["es_anomalia"] == True)).sum()
            TN = ((df_eval["label"] == 0) & (df_eval["es_anomalia"] == False)).sum()

            metricas_iter.append([TP, FN, FP, TN])

        if not metricas_iter:
            continue

        metricas_np = np.array(metricas_iter)
        TP_mean, FN_mean, FP_mean, TN_mean = metricas_np.mean(axis=0)
        TPR = TP_mean / (TP_mean + FN_mean) if (TP_mean + FN_mean) > 0 else 0
        TNR = TN_mean / (TN_mean + FP_mean) if (TN_mean + FP_mean) > 0 else 0
        FPR = FP_mean / (FP_mean + TN_mean) if (FP_mean + TN_mean) > 0 else 0
        rendimiento = np.sqrt(TPR * TNR)

        resultados.append({
            "usuario": usuario,
            "k": k,
            "TP": round(TP_mean),
            "FN": round(FN_mean),
            "FP": round(FP_mean),
            "TN": round(TN_mean),
            "TPR": round(TPR, 4),
            "FPR": round(FPR, 4),
            "TNR": round(TNR, 4),
            "Rendimiento": round(rendimiento, 4)
        })

    df_usuario = pd.DataFrame(resultados)
    resultados_globales.append(df_usuario)

df_total = pd.concat(resultados_globales, ignore_index=True)
df_total.to_csv(ruta_script / f"resultados_todos_usuarios_inyeccion_realista_{'normalizado' if normalizar else 'original'}.csv", sep=";", index=False)

df_media = df_total.groupby("k").mean(numeric_only=True).reset_index()
for col in ["TP", "FN", "FP", "TN"]:
    if col in df_media.columns:
        df_media[col] = df_media[col].round(0).astype(int)

for col in ["TPR", "FPR", "TNR", "Rendimiento"]:
    if col in df_media.columns:
        df_media[col] = df_media[col].round(4)

df_media.to_csv(ruta_script / f"resultados_medias_globales_inyeccion_realista_{'normalizado' if normalizar else 'original'}.csv", sep=";", index=False)

print("\n✅ Evaluación completada con conteo real de anomalías inyectadas.")


