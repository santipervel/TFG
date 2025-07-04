import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Cargar datos ---
ruta_csv = Path(__file__).resolve().parent / "Resultados_EWMA_tipo_evento" / "resumen_global_metricas_por_tipo.csv"
df = pd.read_csv(ruta_csv, sep=";")

# --- Crear carpeta de salida ---
output_dir = ruta_csv.parent / "figuras_EWMA"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Métricas a graficar ---
metricas = ["TPR", "FPR", "rendimiento"]

# --- Generar una figura por métrica, por valor de delta y por valor de k ---
for metrica in metricas:
    for delta_val in sorted(df["delta"].unique()):
        for k_val in sorted(df["k"].unique()):
            df_kd = df[(df["k"] == k_val) & (df["delta"] == delta_val)]
            if df_kd.empty:
                continue

            plt.figure(figsize=(10, 6))
            for alpha_val in sorted(df_kd["alpha"].unique()):
                df_plot = df_kd[df_kd["alpha"] == alpha_val]
                plt.plot(df_plot["window_size"], df_plot[metrica], marker='o', label=f"alpha={alpha_val}")

            plt.title(f"{metrica.upper()} vs Tamaño de ventana\nk={k_val}, delta={delta_val}")
            plt.xlabel("Tamaño de ventana (minutos)")
            plt.ylabel(metrica.upper())
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            nombre_figura = output_dir / f"{metrica}_k{k_val}_delta{delta_val}.png"
            plt.savefig(nombre_figura)
            plt.close()
            print(f"✅ Guardada: {nombre_figura}")
