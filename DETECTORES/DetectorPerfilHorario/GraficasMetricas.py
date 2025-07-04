import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Parámetros ---
ruta_script = Path(__file__).resolve().parent
archivo_resultados = ruta_script / "Perfil_Horario_EventosUsuarios" / "normalizado" / "resultados_medias_globales_normalizado.csv"  # Cambia el nombre
output_dir = ruta_script / "Graficas_2.1"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Cargar datos ---
df = pd.read_csv(archivo_resultados, sep=";")

# Asegurarse de que k es numérico (por si lo lee como texto)
df["k"] = df["k"].astype(float)

# --- Gráfica TPR ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="k", y="TPR", hue="factor_anomalia", marker="o", palette="tab10")
plt.title("TPR medio vs k (una curva por factor)")
plt.xlabel("Valor de k")
plt.ylabel("TPR medio")
plt.grid(True)
plt.legend(title="Factor de anomalía")
plt.tight_layout()
plt.savefig(output_dir / "TPR_vs_k.png")
plt.close()

# --- Gráfica FPR ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="k", y="FPR", hue="factor_anomalia", marker="o", palette="tab10")
plt.title("FPR medio vs k (una curva por factor)")
plt.xlabel("Valor de k")
plt.ylabel("FPR medio")
plt.grid(True)
plt.legend(title="Factor de anomalía")
plt.tight_layout()
plt.savefig(output_dir / "FPR_vs_k.png")
plt.close()

# --- Gráfica Rendimiento ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="k", y="Rendimiento", hue="factor_anomalia", marker="o", palette="tab10")
plt.title("Rendimiento vs k (una curva por factor)")
plt.xlabel("Valor de k")
plt.ylabel("Rendimiento medio")
plt.grid(True)
plt.legend(title="Factor de anomalía")
plt.tight_layout()
plt.savefig(output_dir / "Rendimiento_vs_k.png")
plt.close()

print("✅ Gráficas guardadas en la carpeta 'Graficas_2.1'")
