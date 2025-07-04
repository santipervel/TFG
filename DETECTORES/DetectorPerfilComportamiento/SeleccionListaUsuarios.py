import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# --- Configuración ---
ruta_script = Path(__file__).resolve().parent
ruta_archivos = ruta_script.parent.parent
archivos_2018 = [ruta_archivos / f"split_part_{i}.jsonl" for i in range(59, 143)]

# --- Cargar datos completos de 2018 ---
df = pd.concat([pd.read_json(f, lines=True) for f in archivos_2018])
df["time"] = pd.to_datetime(df["time"])
df = df[(df['time'] >= '2018-01-01') & (df['time'] < '2019-01-01')]
df["fecha"] = df["time"].dt.date

print(f"👥 Total de usuarios únicos en 2018: {df['uid'].nunique()}")


# --- Agrupar por usuario ---
resumen_usuarios = (
    df.groupby("uid")
    .agg(
        eventos_totales=("time", "count"),
        dias_activos=("fecha", "nunique"),
        primer_dia=("fecha", "min"),
        ultimo_dia=("fecha", "max"),
        rol=("role", "first")
    )
    .reset_index()
)

# --- Filtros mínimos ---
usuarios_filtrados = resumen_usuarios[
    (resumen_usuarios["dias_activos"] >= 60) &
    (resumen_usuarios["eventos_totales"] >= 300)
].copy()

# --- Seleccionar los 100 con más eventos (criterio ajustable) ---
usuarios_seleccionados = usuarios_filtrados.sort_values(
    by="eventos_totales", ascending=False
).head(100)

# --- Guardar a CSV ---
output_dir = ruta_script / "usuarios_seleccionados"
output_dir.mkdir(parents=True, exist_ok=True)
usuarios_seleccionados.to_csv(output_dir / "top_100_usuarios_2018.csv", index=False, sep=";")

print(f"✅ Seleccionados {len(usuarios_seleccionados)} usuarios.")
print(usuarios_seleccionados.head())

# --- Crear carpeta para gráficas ---
carpeta_graficas = ruta_script / "graficas_actividad_usuarios"
carpeta_graficas.mkdir(parents=True, exist_ok=True)

# --- Generar una gráfica por usuario ---
uids_seleccionados = usuarios_seleccionados["uid"].tolist()
df = df[df["uid"].isin(uids_seleccionados)]

for _, row in usuarios_seleccionados.iterrows():
    uid = row["uid"]
    rol = row["rol"]

    df_user = df[df["uid"] == uid]
    actividad_diaria = df_user.groupby("fecha").size()

    # Rellenar fechas faltantes con 0
    fechas_2018 = pd.date_range("2018-01-01", "2018-12-31").date
    actividad_completa = pd.Series(0, index=fechas_2018)
    actividad_completa.update(actividad_diaria)

    # Crear gráfica
    plt.figure(figsize=(12, 4))
    plt.plot(actividad_completa.index, actividad_completa.values, color="steelblue")
    plt.title(f"Actividad diaria de {uid}\nRol: {rol}")
    plt.xlabel("Fecha")
    plt.ylabel("Número de eventos")
    plt.grid(True)
    plt.tight_layout()

    # Guardar como PNG
    path_imagen = carpeta_graficas / f"{uid}.png"
    plt.savefig(path_imagen)
    plt.close()

print(f"📁 Gráficas guardadas en: {carpeta_graficas}")


