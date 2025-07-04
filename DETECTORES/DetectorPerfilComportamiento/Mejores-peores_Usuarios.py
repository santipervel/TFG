import pandas as pd
from pathlib import Path

# --- Parámetros ---
ruta_script = Path(__file__).resolve().parent
archivo_resultados = ruta_script / "resultados_todos_usuarios_original.csv"  # Cambiar el nombre 

# --- Leer CSV ---
df = pd.read_csv(archivo_resultados, sep=";")

# --- Filtrar por combinación seleccionada ---
df_filtrado = df[(df["k"] == 0.5) & (df["factor_anomalia"] == 5)]

# --- Ordenar por rendimiento ---
df_ordenado = df_filtrado.sort_values(by="Rendimiento", ascending=False)

# --- Obtener 10 mejores y 10 peores usuarios ---
mejores_10 = df_ordenado.head(15)
peores_10 = df_ordenado.tail(15)

# --- Mostrar resultados ---
print("\n🎯 10 usuarios con mayor rendimiento:\n")
print(mejores_10[["usuario", "TP", "FN", "FP", "TN", "TPR", "FPR", "TNR", "Rendimiento"]])

print("\n🎯 10 usuarios con menor rendimiento:\n")
print(peores_10[["usuario", "TP", "FN", "FP", "TN", "TPR", "FPR", "TNR", "Rendimiento"]])

# --- (Opcional) Guardar en CSV si quieres añadirlo al informe ---
mejores_10.to_csv(ruta_script / "top_10_usuarios.csv", sep=";", index=False)
peores_10.to_csv(ruta_script / "bottom_10_usuarios.csv", sep=";", index=False)

print("\n✅ Archivos 'top_10_usuarios.csv' y 'bottom_10_usuarios.csv' generados.")
