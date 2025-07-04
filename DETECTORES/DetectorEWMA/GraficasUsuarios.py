import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Parámetros del usuario y combinación ---
usuario = "selected-beige-vole-recorder"
window_size = 5
delta = 3
alpha = 0.6
k = 0.5

# --- Rutas locales ---
ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]

output_dir = ruta_script / "Resultados_EWMA"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Cargar datos y filtrar ---
df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])
df["time"] = pd.to_datetime(df["time"])
df = df[(df["time"] >= "2018-07-01") & (df["time"] < "2019-01-01")]
df_usuario = df[df["uid"] == usuario].copy()
df_usuario = df_usuario.sort_values("time").set_index("time")

# --- Agrupar en ventanas ---
df_agg = df_usuario.resample(f"{window_size}min").size().reset_index(name="eventos")

# --- Calcular EWMA hasta t-1 ---
eventos_shift = df_agg["eventos"].shift(1)
df_agg["ewma"] = eventos_shift.ewm(alpha=alpha, adjust=False).mean()
df_agg["std"] = eventos_shift.expanding().std()
df_agg["umbral_sup"] = df_agg["ewma"] + k * df_agg["std"]
df_agg["anomalía"] = df_agg["eventos"] > df_agg["umbral_sup"]

# --- Gráfica ---
plt.figure(figsize=(15, 6))
plt.scatter(df_agg["time"], df_agg["eventos"], label="Eventos", color='blue', s=10)
plt.plot(df_agg["time"], df_agg["ewma"], label="EWMA (hasta t-1)", color='orange')
plt.plot(df_agg["time"], df_agg["umbral_sup"], label="Umbral superior", color='red', linestyle='--')

anomalias = df_agg[df_agg["anomalía"]]
plt.scatter(anomalias["time"], anomalias["eventos"], color='red', label="Anomalías", s=30, zorder=5)

plt.title(f"EWMA para {usuario} (win={window_size}min, α={alpha}, k={k})")
plt.xlabel("Tiempo")
plt.ylim(0, 100)
plt.ylabel("Nº de eventos por ventana")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
archivo = output_dir / f"grafica_{usuario}_win{window_size}_alpha{alpha}_k{k}_prueba.png"
plt.savefig(archivo)
