import pandas as pd
from pathlib import Path

# --- Configuración ---
ruta_script = Path(__file__).resolve().parent
archivo_resultados = ruta_script / "Perfil_Horario_EventosUsuarios" / "normalizado" / "metricas_inyeccion_k_0.5_factor_12.0.csv"

# --- Leer CSV ---
df = pd.read_csv(archivo_resultados, sep=";")

# --- Asegurar tipos numéricos ---
columnas_float = ["TPR", "FPR", "TNR", "Rendimiento"]
columnas_int = ["TP", "FN", "FP", "TN"]

for col in columnas_float:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in columnas_int:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# --- Calcular promedio de rendimiento por usuario ---
df_usuario = df.groupby("usuario").mean(numeric_only=True).reset_index()

# --- Ordenar por rendimiento medio ---
df_ordenado = df_usuario.sort_values(by="Rendimiento", ascending=False)

# --- Obtener top 10 y bottom 10 ---
mejores_10 = df_ordenado.head(10)
peores_10 = df_ordenado.tail(10)

# --- Mostrar resultados ---
print("\n🎯 10 usuarios con mayor rendimiento medio:\n")
print(mejores_10[["usuario", "TP", "FN", "FP", "TN", "TPR", "FPR", "TNR", "Rendimiento"]])

print("\n🎯 10 usuarios con menor rendimiento medio:\n")
print(peores_10[["usuario", "TP", "FN", "FP", "TN", "TPR", "FPR", "TNR", "Rendimiento"]])

# --- Guardar en CSV ---
mejores_10.to_csv(ruta_script / "top_10_usuarios.csv", sep=";", index=False, float_format="%.4f" )
peores_10.to_csv(ruta_script / "bottom_10_usuarios.csv", sep=";", index=False, float_format="%.4f")

print("\n✅ Archivos 'top_10_usuarios.csv' y 'bottom_10_usuarios.csv' generados.")
