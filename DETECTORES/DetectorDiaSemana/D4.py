import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import setup, create_model, predict_model
from sklearn.ensemble import IsolationForest  

# === Parámetros globales ===
ITERACIONES = 30
normalizar = True
graficar = False
modelo = "knn"  # Modelo a usar
factor_anomalia = 2.0
num_anomalias = 35


top_10_mejores = ['little-apricot-baboon-calibrationmanager', 'precise-coffee-dog-locksmith', 'well-orange-spoonbill-icecreamvendor', 'vast-fuchsia-lemur-screenprinter', 'varying-brown-nightingale-book-keeper', 'frightened-emerald-sloth-magistrate', 'fresh-lime-perch-projectengineer', 'ethical-lavender-clownfish-stagemover', 'yellow-chocolate-carp-busdriver', 'statutory-silver-ocelot-warehousemanager']
top_10_peores = ['green-white-frog-chimneysweep', 'communist-indigo-possum-forester', 'integral-turquoise-narwhal-rugmaker', 'strong-moccasin-pony-pipeinspector', 'accurate-tan-mandrill-qualityengineer', 'lively-pink-narwhal-yachtmaster', 'shared-fuchsia-cardinal-buildingadvisor', 'salty-fuchsia-dragon-machinist', 'victorious-gray-clam-taxadvisor', 'big-maroon-lynx-radiocontroller'] 

# === Rutas ===
ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
ruta_usuarios = ruta_script.parent / "DetectorPerfilComportamiento" / "usuarios_seleccionados" / "top_100_usuarios_2018.csv"
output_dir = ruta_script.parent / "DetectorDiaSemana" / "Resultados_Inyeccion_2D"
output_dir.mkdir(parents=True, exist_ok=True)

# === Funciones auxiliares ===
def crear_calendario(inicio, fin):
    fechas = pd.date_range(inicio, fin, freq="D")
    calendario = pd.DataFrame({"fecha": fechas})
    calendario["dia_semana"] = calendario["fecha"].dt.day_name()
    calendario["dia_codificado"] = calendario["dia_semana"].map({
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    })
    calendario["fecha"] = calendario["fecha"].dt.date
    return calendario

def generar_dataset(df_user, normalizar):
    df_train = df_user[df_user["time"] < "2018-07-01"].copy()
    calendario = crear_calendario("2018-01-01", "2018-06-30")
    df_grouped = df_train.groupby("fecha").size().reset_index(name="eventos")
    df_completo = pd.merge(calendario, df_grouped, how="left", on="fecha").fillna(0)
    
    media, std = df_completo["eventos"].mean(), df_completo["eventos"].std()
    if normalizar:
        df_completo["eventos"] = (df_completo["eventos"] - media) / std

    clase_0 = df_completo[["dia_codificado", "eventos"]].copy()
    clase_0["clase"] = 0
    max_eventos = clase_0["eventos"].max()
    
    puntos_clase_1 = []

    for dia in range(7):
        eventos_dia = clase_0[clase_0["dia_codificado"] == dia]["eventos"]
        eventos_filtrados = eventos_dia if normalizar else eventos_dia[eventos_dia > 1]

        if eventos_filtrados.empty:
            continue

        p15, p85 = np.percentile(eventos_filtrados, [15, 85])
        eventos_validos = eventos_filtrados[(eventos_filtrados >= p15) & (eventos_filtrados <= p85)]
        if eventos_validos.empty:
            continue

        min_ev, max_ev = eventos_validos.min(), eventos_validos.max()
        max_eventos = clase_0["eventos"].max()

        # --- Generar puntos bajos SOLO si no está normalizado ---
        if not normalizar and min_ev > 1:
            inicio = min_ev - 0.6 * (min_ev - 1)
            puntos_bajo = np.random.uniform(inicio, min_ev, 10)
            puntos_clase_1.extend([(dia, y) for y in puntos_bajo])

        # --- Generar puntos altos ---
        if normalizar:
            # Separación clara respecto a valores normales cercanos a 0
            puntos_alto = np.random.uniform(0.5, max_eventos, 40)
        else:
            if max_ev < max_eventos:
                inicio = max_ev + 0.01 * (max_eventos - max_ev)
                puntos_alto = np.random.uniform(inicio, max_eventos, 40)
            else:
                continue  # No hay espacio para generar puntos altos

        puntos_clase_1.extend([(dia, y) for y in puntos_alto])



    clase_1 = pd.DataFrame(puntos_clase_1, columns=["dia_codificado", "eventos"])
    clase_1["clase"] = 1
    return pd.concat([clase_0, clase_1]).reset_index(drop=True), media, std, clase_0
    return pd.concat([clase_0, clase_1]).reset_index(drop=True), media, std
def evaluar_modelo(modelo, df_test, media, std, usuario):
    calendario_test = crear_calendario("2018-07-01", "2018-12-31")
    df_test_grouped = df_test.groupby("fecha").size().reset_index(name="eventos")
    df_test_completo = pd.merge(calendario_test, df_test_grouped, how="left", on="fecha").fillna(0)
    if normalizar:
        df_test_completo["eventos"] = (df_test_completo["eventos"] - media) / std
    df_test_ready = df_test_completo[["dia_codificado", "eventos"]].copy()
    df_inyectado = df_test_ready.copy()
    df_inyectado["label"] = 0
    np.random.seed(42)
    indices = df_inyectado.sample(n=num_anomalias).index
    if normalizar:
        df_inyectado.loc[indices, "eventos"] += factor_anomalia
    else:
        df_inyectado.loc[indices, "eventos"] += media * factor_anomalia
    df_inyectado.loc[indices, "label"] = 1
    df_pred = predict_model(modelo, data=df_inyectado.copy())
    df_pred["real"] = df_inyectado["label"]
    df_pred["pred"] = df_pred["prediction_label"]
    TP = ((df_pred["real"] == 1) & (df_pred["pred"] == 1)).sum()
    FN = ((df_pred["real"] == 1) & (df_pred["pred"] == 0)).sum()
    FP = ((df_pred["real"] == 0) & (df_pred["pred"] == 1)).sum()
    TN = ((df_pred["real"] == 0) & (df_pred["pred"] == 0)).sum()
    TPR = TP / (TP + FN) if (TP + FN) else 0
    TNR = TN / (TN + FP) if (TN + FP) else 0
    FPR = FP / (FP + TN) if (FP + TN) else 0
    rendimiento = np.sqrt(TPR * TNR)
    print(f"\n📊 [{usuario}] TP={TP} FN={FN} FP={FP} TN={TN} → TPR={TPR:.3f}, TNR={TNR:.3f}, Rend={rendimiento:.3f}")
    return TP, FN, FP, TN, TPR, TNR, FPR, rendimiento, df_pred

# === Carga de datos ===
if graficar:
    usuarios_total = top_10_mejores + top_10_peores  # Usar los usuarios seleccionados 
usuarios_total = pd.read_csv(ruta_usuarios, sep=";")["uid"].tolist()
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]
df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])
df["time"] = pd.to_datetime(df["time"])
df["fecha"] = df["time"].dt.date
df["dia_semana"] = df["time"].dt.day_name()
df["dia_codificado"] = df["dia_semana"].map({
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
})

resultados = []

# === Iteraciones principales ===
for iteracion in range(ITERACIONES):
    print(f"\n🔁 Iteración {iteracion + 1}/{ITERACIONES}")
    for usuario in usuarios_total:
        print(f"\n🔍 Procesando usuario: {usuario}")
        try:
            df_user = df[df["uid"] == usuario].copy()
            df_final, media, std, puntos_entrenamiento = generar_dataset(df_user, normalizar)


            if graficar: 
                # --- Visualización ---
                df_viz = df_final.copy()
                if not normalizar :
                    df_viz["eventos"] += 1
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df_viz, x="dia_codificado", y="eventos", hue="clase", palette={0: "blue", 1: "red"}, alpha=0.6)
                if not normalizar:
                    plt.yscale("log")
                plt.xlabel("Día de la semana (0=Lunes)")
                plt.ylabel("Eventos (log)")
                plt.title(f"{usuario} - Distribución de eventos")
                plt.grid(True)
                plt.tight_layout()
                carpeta_usuario = "Top_10_Mejores" if usuario in top_10_mejores else "Top_10_Peores"
                ruta_img = ruta_script.parent / "DetectorDiaSemana" / "Graficas_2D" / "Graficas_KNN_Normalizado" / carpeta_usuario
                ruta_img.mkdir(parents=True, exist_ok=True)
                plt.savefig(ruta_img / f"{usuario}_KNN_Normalizado.png")
                plt.close()

            # --- Entrenamiento ---
            setup(df_final, target="clase", session_id=42, verbose=False)
            modelo = create_model(modelo)

            # --- Evaluación ---
            df_test = df_user[df_user["time"] >= "2018-07-01"].copy()
            TP, FN, FP, TN, TPR, TNR, FPR, rendimiento, df_pred = evaluar_modelo(modelo, df_test, media, std, usuario)
            if graficar:
                df_pred["tipo"] = "TN"
                df_pred.loc[(df_pred["real"] == 1) & (df_pred["pred"] == 1), "tipo"] = "TP"
                df_pred.loc[(df_pred["real"] == 1) & (df_pred["pred"] == 0), "tipo"] = "FN"
                df_pred.loc[(df_pred["real"] == 0) & (df_pred["pred"] == 1), "tipo"] = "FP"
                puntos_entrenamiento_plot = puntos_entrenamiento.copy()
                puntos_entrenamiento_plot["tipo"] = "Train"

                df_plot = pd.concat([puntos_entrenamiento_plot[["dia_codificado", "eventos", "tipo"]], df_pred[["dia_codificado", "eventos", "tipo"]]])
                #df_plot["dia_codificado_jitter"] = df_plot["dia_codificado"] + np.random.uniform(-0.05, 0.05, size=len(df_plot))

                plt.figure(figsize=(8, 6))
                palette = {
                    "Train": "gray",
                    "TN": "blue",
                    "TP": "green",
                    "FP": "red",
                    "FN": "orange"
                }
                sns.scatterplot(data=df_plot, x="dia_codificado", y="eventos", hue="tipo", palette=palette, alpha=0.6)

                plt.xlabel("Día de la semana")
                plt.ylabel("Eventos")
                plt.title(f"{usuario} - Evaluación 5 colores")
                plt.grid(True)
                plt.tight_layout()

                # Directorio base
                base_dir = ruta_script.parent / "DetectorDiaSemana" / "Graficas_2D" / "Evaluacion_5_colores"
                dir_mejores = base_dir / "Mejores"
                dir_peores = base_dir / "Peores"

                # Crear directorios si no existen
                dir_mejores.mkdir(parents=True, exist_ok=True)
                dir_peores.mkdir(parents=True, exist_ok=True)

                # Decidir carpeta según la lista
                if usuario in top_10_mejores:
                    ruta_guardado = dir_mejores / f"{usuario}_evaluacion_5colores.png"
                elif usuario in top_10_peores:
                    ruta_guardado = dir_peores / f"{usuario}_evaluacion_5colores.png"

                plt.savefig(ruta_guardado)
                plt.close()


            resultados.append({
                "usuario": usuario, "iteracion": iteracion + 1,
                "TP": TP, "FN": FN, "FP": FP, "TN": TN,
                "TPR": round(TPR, 4), "TNR": round(TNR, 4),
                "FPR": round(FPR, 4), "Rendimiento": round(rendimiento, 4)
            })
        except Exception as e:
            print(f"⚠️ Error con {usuario}: {e}")

# === Guardar resultados ===
df_resultados = pd.DataFrame(resultados)
df_media = df_resultados.groupby("usuario").agg(
    TP=("TP", "mean"), FN=("FN", "mean"), FP=("FP", "mean"),
    TN=("TN", "mean"), TPR=("TPR", "mean")
).reset_index()
df_media["FPR"] = df_media["FP"] / (df_media["FP"] + df_media["TN"])
df_media["TNR"] = df_media["TN"] / (df_media["FP"] + df_media["TN"])
df_media["Rendimiento"] = np.sqrt(df_media["TPR"] * df_media["TNR"])
df_media.to_csv(output_dir / "metricas_KNN_normalizado.csv", sep=";", index=False)
print("\n✅ Métricas guardadas en 'metricas_KNN_normalizado.csv'")
