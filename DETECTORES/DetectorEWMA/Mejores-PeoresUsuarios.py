import pandas as pd
from pathlib import Path
# --- Parámetros elegidos ---
window_size_objetivo = 5
alpha_objetivo = 0.6
k_objetivo = 0.5
delta_objetivo = 3

# --- Cargar CSV ---
ruta_script = Path(__file__).resolve().parent
ruta_csv = ruta_script.parent / "DetectorEWMA" / "Resultados_EWMA" / "resumen_metricas_por_usuario_delta3.csv"  # cambia si es otro nombre

df = pd.read_csv(ruta_csv, sep=";")

# --- Filtrar por parámetros ---
df_filtrado = df[
    (df["window_size"] == window_size_objetivo) &
    (df["alpha"] == alpha_objetivo) &
    (df["k"] == k_objetivo) &
    (df["delta"] == delta_objetivo)
]

# --- Ordenar por rendimiento ---
top_mejores = df_filtrado.sort_values("rendimiento", ascending=False).head(10)
top_peores = df_filtrado.sort_values("rendimiento", ascending=True).head(10)

# --- Mostrar resultados ---
print("🔝 Top 10 usuarios con mejor rendimiento:")
print(top_mejores[["usuario", "rendimiento", "TPR", "FPR", "TNR"]])

print("\n🔻 Top 10 usuarios con peor rendimiento:")
print(top_peores[["usuario", "rendimiento", "TPR", "FPR", "TNR"]])


# --- Parámetros elegidos segundo detector ---
window_size_objetivo = 5
alpha_objetivo = 0.3
k_objetivo = 0.3
delta_objetivo = 3

# --- Cargar CSVs subidos ---
df_usuarios = ruta_script.parent / "DetectorEWMA" / "Resultados_EWMA" / "resumen_metricas_por_usuario_delta3.csv"  # cambia si es otro nombre
df_usuarios = pd.read_csv(df_usuarios, sep=";")
ruta_csv = ruta_script.parent / "DetectorEWMA" / "Resultados_EWMA_tipo_evento" / "resumen_metricas_por_usuario_y_tipo.csv"  # cambia si es otro nombre

df_tipos = ruta_script.parent / "DetectorEWMA" / "Resultados_EWMA_tipo_evento" / "resumen_metricas_por_tipo.csv"  # cambia si es otro nombre
df_tipos = pd.read_csv(df_tipos, sep=";")
# --- Filtrar por parámetros ---
df_usuarios_filtro = df_usuarios[
    (df_usuarios["window_size"] == window_size_objetivo) &
    (df_usuarios["alpha"] == alpha_objetivo) &
    (df_usuarios["k"] == k_objetivo) &
    (df_usuarios["delta"] == delta_objetivo)
]

df_tipos_filtro = df_tipos[
    (df_tipos["window_size"] == window_size_objetivo) &
    (df_tipos["alpha"] == alpha_objetivo) &
    (df_tipos["k"] == k_objetivo) &
    (df_tipos["delta"] == delta_objetivo)
]

# --- Agrupar por usuario (media sobre tipos de evento) ---
df_rendimiento_usuarios = df_usuarios_filtro.groupby("usuario")[["TPR", "FPR", "TNR", "rendimiento"]].mean().reset_index()

# --- Top 10 usuarios con mejor y peor rendimiento ---
top_10_mejores = df_rendimiento_usuarios.sort_values("rendimiento", ascending=False).head(10)
top_10_peores = df_rendimiento_usuarios.sort_values("rendimiento", ascending=True).head(10)

# --- Métricas medias por tipo de evento ---
df_metricas_tipos = df_tipos_filtro.groupby("evento")[["TPR", "FPR", "TNR", "rendimiento"]].mean().reset_index()

# --- Mostrar resultados ---
print("🔝 Top 10 usuarios con mejor rendimiento:")
print(top_10_mejores)

print("\n🔻 Top 10 usuarios con peor rendimiento:")
print(top_10_peores)

print("\n📊 Métricas medias por tipo de evento:")
print(df_metricas_tipos)
