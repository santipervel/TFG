import pandas as pd
import numpy as np
from pathlib import Path
from pycaret.anomaly import setup, create_model, predict_model

# --- Configuración ---
NUM_ITERACIONES = 30
ANOMALIAS_POR_DIA = 5
FACTOR_AUMENTO = 2
# modelos = ["iforest", "lof"]
# normalizaciones = [True, False]

#Combinación óptima
modelos = ["lof"]
normalizaciones = ["False"]

contamination = 0.05

ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]
ruta_usuarios = ruta_script.parent / "DetectorPerfilComportamiento" / "usuarios_seleccionados" / "top_100_usuarios_2018.csv"
output_dir = ruta_script.parent / "DetectorDiaSemana" / "Resultados_Inyeccion"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Cargar lista de usuarios ---
usuarios_df = pd.read_csv(ruta_usuarios, sep=";")
usuarios_lista = usuarios_df["uid"].tolist()

# --- Cargar dataset completo ---
df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])
df["time"] = pd.to_datetime(df["time"])
df = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")]

# --- Inicializar resultados globales ---
resultados_tp_fn = []

for modelo in modelos:
    for normalizar in normalizaciones:
        for usuario in usuarios_lista:
            print(f"\n🔵 Procesando usuario: {usuario} | Modelo: {modelo} | Normalizar: {normalizar}")
            try:
                df_user = df[df["uid"] == usuario].copy()
                df_user["fecha"] = df_user["time"].dt.date
                df_user["dia_semana"] = df_user["time"].dt.day_name()

                df_entrenamiento = df_user[df_user["time"] < "2018-07-01"].copy()
                df_test_base = df_user[df_user["time"] >= "2018-07-01"].copy()

                fechas_entrenamiento = pd.date_range("2018-01-01", "2018-06-30", freq="D")
                calendario_entrenamiento = pd.DataFrame({"fecha": fechas_entrenamiento})
                calendario_entrenamiento["dia_semana"] = calendario_entrenamiento["fecha"].dt.day_name()
                calendario_entrenamiento["fecha"] = calendario_entrenamiento["fecha"].dt.date

                df_agrupado_entrenamiento = df_entrenamiento.groupby(["fecha", "dia_semana"]).size().reset_index(name="eventos")
                df_entrenamiento_final = pd.merge(calendario_entrenamiento, df_agrupado_entrenamiento, how="left", on=["fecha", "dia_semana"])
                df_entrenamiento_final["eventos"] = df_entrenamiento_final["eventos"].fillna(0).astype(int)

                media_eventos = df_entrenamiento_final["eventos"].mean()
                desviacion_eventos = df_entrenamiento_final["eventos"].std()

                if normalizar:
                    df_entrenamiento_final["eventos"] = (df_entrenamiento_final["eventos"] - media_eventos) / desviacion_eventos

                for iteracion in range(NUM_ITERACIONES):
                    print(f"  🔄 Iteración {iteracion + 1}/{NUM_ITERACIONES}")

                    for dia in df_test_base["dia_semana"].unique():
                        df_dia_test = df_test_base[df_test_base["dia_semana"] == dia].copy()
                        df_dia_entrenamiento = df_entrenamiento_final[df_entrenamiento_final["dia_semana"] == dia].copy()

                        calendario_test = pd.date_range("2018-07-01", "2018-12-31", freq="D")
                        calendario_test = pd.DataFrame({"fecha": calendario_test})
                        calendario_test["dia_semana"] = calendario_test["fecha"].dt.day_name()
                        calendario_test["fecha"] = calendario_test["fecha"].dt.date

                        df_agrupado_test = df_dia_test.groupby("fecha").size().reset_index(name="eventos")
                        df_completo_test = pd.merge(calendario_test, df_agrupado_test, how="left", on="fecha")
                        df_completo_test["eventos"] = df_completo_test["eventos"].fillna(0).astype(int)
                        df_completo_test = df_completo_test[df_completo_test["dia_semana"] == dia].copy()

                        if df_completo_test.empty or df_completo_test["eventos"].nunique() <= 1 or df_dia_entrenamiento.empty:
                            continue

                        num_anomalias_real = min(ANOMALIAS_POR_DIA, len(df_completo_test))
                        anomalas = df_completo_test.sample(n=num_anomalias_real, random_state=iteracion + 42).copy()
                        anomalas_indices = anomalas.index.tolist()

                        df_completo_test["label"] = 0
                        df_completo_test["eventos_original"] = df_completo_test["eventos"]
                        df_completo_test.loc[anomalas_indices, "eventos"] = (
                            df_completo_test.loc[anomalas_indices, "eventos"] + FACTOR_AUMENTO * media_eventos
                        ).round().astype(int)
                        df_completo_test.loc[anomalas_indices, "label"] = 1

                        if normalizar:
                            df_completo_test["eventos"] = (df_completo_test["eventos"] - media_eventos) / desviacion_eventos

                        columna_usada = "eventos"

                        try:
                            setup(
                                data=df_dia_entrenamiento[[columna_usada]].rename(columns={columna_usada: "eventos"}),
                                session_id=iteracion,
                                verbose=False
                            )

                            model = create_model(modelo, contamination=contamination)

                            df_predicciones = predict_model(
                                model,
                                data=df_completo_test[[columna_usada]].rename(columns={columna_usada: "eventos"})
                            )

                            TP = ((df_completo_test["label"] == 1) & (df_predicciones["Anomaly"] == 1)).sum()
                            FN = ((df_completo_test["label"] == 1) & (df_predicciones["Anomaly"] == 0)).sum()
                            FP = ((df_completo_test["label"] == 0) & (df_predicciones["Anomaly"] == 1)).sum()
                            TN = ((df_completo_test["label"] == 0) & (df_predicciones["Anomaly"] == 0)).sum()

                            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                            #rendimiento = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

                            resultados_tp_fn.append({
                                "usuario": usuario,
                                "modelo": modelo,
                                "normalizado_manual": normalizar,
                                "iteracion": iteracion + 1,
                                "dia_semana": dia,
                                "contamination": contamination,
                                "TP": TP,
                                "FN": FN,
                                "FP": FP,
                                "TN": TN,
                                "TPR": round(TPR, 4),
                                #"Rendimiento": round(rendimiento, 4)
                            })

                        except Exception as e:
                            print(f"⚠️ Error en {usuario}, día {dia}, iteración {iteracion + 1}: {e}")

            except Exception as e:
                print(f"❌ Error general con usuario {usuario}: {e}")

# --- Guardar resultados ---
df_final = pd.DataFrame(resultados_tp_fn)
ruta_salida = output_dir / "tp_fn_inyeccion_todos_usuarios_iforest_lof2.csv"
# df_final.to_csv(ruta_salida, index=False, sep=";")
print(f"\n✅ Resultados guardados en {ruta_salida}")
