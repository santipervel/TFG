import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuración ---
usuario = "spectacular-copper-cheetah-postman"
ruta_script = Path(__file__).resolve().parent
ruta_resultados = ruta_script.parent / "DetectorDiaSemana" / "Resultados_Inyeccion"

# --- Cargar los CSV generados ---
print("\U0001F4C5 Cargando datos de anomalías...")
df_anomalias = pd.read_csv(ruta_resultados / f"anomalas_inyectadas_detectadas_{usuario}.csv", sep=";")
df_test_original = pd.read_csv(ruta_script.parent / "DetectorDiaSemana" / "Datos_Preparados" / f"datos_test_{usuario}.csv", sep=";")

# --- Definiciones ---
modelos = ["iforest", "lof", "knn", "svm", "pca"]
normalizaciones = [True, False]
contamination = 0.05

# --- Crear carpeta de gráficas ---
output_graficas = ruta_resultados / "Graficas_Anomalias_Subplots"
output_graficas.mkdir(parents=True, exist_ok=True)

print("\n\U0001F680 Generando gráficas de anomalías (dos subplots)...")

for dia in df_test_original["dia_semana"].unique():
    df_dia_original = df_test_original[df_test_original["dia_semana"] == dia].copy()
    df_anomalias_dia = df_anomalias[df_anomalias["dia_semana"] == dia]

    if df_dia_original.empty or df_anomalias_dia.empty:
        continue

    # Construir el df modificado manualmente
    df_dia_modificado = df_dia_original.copy()
    for idx, row in df_anomalias_dia.iterrows():
        fecha = row["fecha"]
        df_dia_modificado.loc[df_dia_modificado["fecha"] == fecha, "eventos"] = row["eventos"]

    fechas = pd.to_datetime(df_dia_original["fecha"])

    for normalizar in normalizaciones:
        for modelo in modelos:
            try:
                columna_detect = f"detected_{modelo}_norm_{normalizar}_cont_{contamination}"

                if columna_detect not in df_anomalias_dia.columns:
                    continue

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

                # Subplot 1: eventos originales
                ax1.plot(fechas, df_dia_original["eventos"], color="blue")
                ax1.set_title(f"Eventos originales - {dia}")
                ax1.set_ylabel("Número de eventos")
                ax1.grid(True)

                # Subplot 2: eventos modificados
                ax2.plot(fechas, df_dia_modificado["eventos"], color="orange")

                # Añadir anomalías detectadas y no detectadas
                detectadas = df_anomalias_dia[df_anomalias_dia[columna_detect] == 1]
                no_detectadas = df_anomalias_dia[df_anomalias_dia[columna_detect] == 0]

                if not detectadas.empty:
                    ax2.scatter(pd.to_datetime(detectadas["fecha"]), detectadas["eventos"], color="green", label="Anomalía detectada", s=50)
                if not no_detectadas.empty:
                    ax2.scatter(pd.to_datetime(no_detectadas["fecha"]), no_detectadas["eventos"], color="red", label="Anomalía NO detectada", s=50)

                ax2.set_title(f"Eventos modificados e inyectados - {dia}")
                ax2.set_xlabel("Fecha")
                ax2.set_ylabel("Número de eventos")
                ax2.legend()
                ax2.grid(True)

                plt.tight_layout()
                nombre_grafica = f"{usuario}_{dia}_{modelo}_norm_{normalizar}.png"
                plt.savefig(output_graficas / nombre_grafica)
                plt.close()

            except Exception as e:
                print(f"⚠️ Error generando gráfica para {modelo}, normalizar={normalizar}: {e}")

print("\n✅ Todas las gráficas generadas en:", output_graficas)
