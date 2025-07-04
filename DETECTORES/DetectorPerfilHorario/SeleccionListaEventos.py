import pandas as pd
from pathlib import Path

# --- Configuración ---
ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]

df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])

# --- Filtrar eventos de 2018 ---
df["time"] = pd.to_datetime(df["time"])
df = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")]
df_train = df[df["time"] < "2018-07-01"]


# --- Contar eventos por tipo ---
conteo_eventos = df["type"].value_counts().reset_index()
conteo_eventos.columns = ["evento", "total_ocurrencias"]
conteo_entrenamiento = df_train["type"].value_counts().reset_index()


# --- Mostrar los 20 más frecuentes ---
print("\nEventos más frecuentes en 2018:")
print(conteo_eventos)
print("\nEventos más frecuentes en el entrenamiento:")
print(conteo_entrenamiento)