
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuración ---
ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]

usuarios = [
    "big-maroon-lynx-radiocontroller",
    "shared-fuchsia-cardinal-buildingadvisor"
]

k = 0.5
factor = 12.0
num_anomalias = 50
normalizar = False
np.random.seed(42)

# --- Cargar datos ---
df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])
df["time"] = pd.to_datetime(df["time"])
df = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")]
df["fecha"] = df["time"].dt.date
df["hora"] = df["time"].dt.hour

todos_resultados = []

for usuario in usuarios:
    df_user = df[df["uid"] == usuario].copy()
    if df_user.empty:
        continue

    df_train = df_user[df_user["time"] < "2018-07-01"]
    df_test = df_user[df_user["time"] >= "2018-07-01"]

    if df_train.empty or df_test.empty:
        continue

    fechas_train = pd.date_range("2018-01-01", "2018-06-30", freq="H")
    fechas_test = pd.date_range("2018-07-01", "2018-12-31", freq="H")

    df_base_train = pd.DataFrame({"time": fechas_train})
    df_base_train["fecha"] = df_base_train["time"].dt.date
    df_base_train["hora"] = df_base_train["time"].dt.hour

    eventos_train = df_train.groupby(["fecha", "hora"]).size().reset_index(name="eventos")
    df_completo_train = pd.merge(df_base_train, eventos_train, how="left", on=["fecha", "hora"])
    df_completo_train["eventos"] = df_completo_train["eventos"].fillna(0)

    perfil = df_completo_train.groupby("hora").agg(
        media_eventos=("eventos", "mean"),
        std_eventos=("eventos", "std")
    ).reset_index()

    df_base_test = pd.DataFrame({"time": fechas_test})
    df_base_test["fecha"] = df_base_test["time"].dt.date
    df_base_test["hora"] = df_base_test["time"].dt.hour

    eventos_test = df_test.groupby(["fecha", "hora"]).size().reset_index(name="eventos")
    df_completo_test = pd.merge(df_base_test, eventos_test, how="left", on=["fecha", "hora"])
    df_completo_test["eventos"] = df_completo_test["eventos"].fillna(0)
    df_completo_test["label"] = 0

    # Inyección de anomalías
    seleccionadas = df_completo_test.sample(n=num_anomalias).copy()
    seleccionadas = pd.merge(seleccionadas, perfil, on="hora", how="left")
    seleccionadas["eventos"] = seleccionadas["eventos"] + seleccionadas["media_eventos"] * factor
    seleccionadas["label"] = 1

    df_completo_test.loc[seleccionadas.index, "eventos"] = seleccionadas["eventos"]
    df_completo_test.loc[seleccionadas.index, "label"] = 1

    df_eval = pd.merge(df_completo_test, perfil, on="hora", how="left")
    df_eval["umbral"] = df_eval["media_eventos"] + k * df_eval["std_eventos"]
    df_eval["pred"] = (df_eval["eventos"] > df_eval["umbral"]).astype(int)

    # Clasificación TP, TN, FP, FN
    df_eval["tipo"] = "TN"
    df_eval.loc[(df_eval["label"] == 1) & (df_eval["pred"] == 1), "tipo"] = "TP"
    df_eval.loc[(df_eval["label"] == 1) & (df_eval["pred"] == 0), "tipo"] = "FN"
    df_eval.loc[(df_eval["label"] == 0) & (df_eval["pred"] == 1), "tipo"] = "FP"

    df_eval["usuario"] = usuario
    todos_resultados.append(df_eval)

# Guardar resultados
df_final = pd.concat(todos_resultados, ignore_index=True)
df_final.to_csv(ruta_script / "detector_2_1_detalle_por_usuario.csv", sep=";", index=False)
print("\n✅ Datos exportados a 'detector_2_1_detalle_por_usuario.csv'")

# === Generación de gráficas de evaluación por usuario ===
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ruta de salida para las gráficas
output_path = ruta_script.parent / "DetectorPerfilHorario" / "Graficas_2.1" 
output_path.mkdir(parents=True, exist_ok=True)

# Leer el CSV generado si no lo tienes ya en memoria
df_detalle = pd.read_csv(ruta_script / "detector_2_1_detalle_por_usuario.csv", sep=";")

# Crear gráficas para cada usuario
usuarios = df_detalle["usuario"].unique()
for usuario in usuarios:
    df_usuario = df_detalle[df_detalle["usuario"] == usuario].copy()

    plt.figure(figsize=(12, 6))
    palette = {"TP": "green", "FP": "red", "TN": "blue", "FN": "orange"}
    sns.scatterplot(
        data=df_usuario,
        x="hora",
        y="eventos",
        hue="tipo",
        palette=palette,
        alpha=0.7
    )

    plt.title(f"Evaluación del detector 2.1 para {usuario}")
    plt.xlabel("Hora del día")
    plt.ylabel("Número de eventos")
    plt.grid(True)
    plt.tight_layout()

    # Guardar la imagen
    nombre_archivo = f"{usuario}_grafica_evaluacion.png"
    plt.savefig(output_path / nombre_archivo)
    plt.close()
