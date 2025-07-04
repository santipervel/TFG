import subprocess
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
PYTHON_ENV = "C:/Users/USUARIO/OneDrive/Escritorio/TFG/Codigo/dataset_venv/Scripts/python.exe"
RUTA_SCRIPT_DETECTOR = "../DetectorDiaSemana/D3.py"
DETECTOR_ID = "D3"
NOMBRE_CSV = "monitor_d3.csv"
CARPETA_GRAFICAS = "graficas"

# --- FUNCIONES ---
def lanzar_detector(script_path):
    """Lanza el detector con el entorno virtual indicado"""
    return subprocess.Popen([PYTHON_ENV, script_path])

def monitor_recursos(nombre_csv, pid_objetivo, intervalo=0.5):
    """Monitoriza RAM y tiempo de CPU total acumulado"""
    datos = []
    try:
        proceso_principal = psutil.Process(pid_objetivo)
        t0 = time.time()

        while proceso_principal.is_running() and proceso_principal.status() != psutil.STATUS_ZOMBIE:
            t_actual = time.time()
            tiempo_transcurrido = t_actual - t0

            # Tiempo de CPU acumulado del proceso principal
            cpu_time = 0.0
            ram_total = 0

            if proceso_principal.is_running():
                try:
                    cpu_times = proceso_principal.cpu_times()
                    cpu_time += cpu_times.user + cpu_times.system
                    ram_total += proceso_principal.memory_info().rss
                except psutil.NoSuchProcess:
                    break

                # Sumar tiempos y RAM de hijos
                for hijo in proceso_principal.children(recursive=True):
                    if hijo.is_running():
                        try:
                            cpu_times = hijo.cpu_times()
                            cpu_time += cpu_times.user + cpu_times.system
                            ram_total += hijo.memory_info().rss
                        except psutil.NoSuchProcess:
                            continue

            datos.append([tiempo_transcurrido, ram_total / 1024**2, cpu_time])  # RAM en MB

            time.sleep(intervalo)

    except psutil.NoSuchProcess:
        print("❌ Proceso terminado inesperadamente.")

    df = pd.DataFrame(datos, columns=["tiempo_s", "RAM_MB", "CPU_acumulado_s"])
    df.to_csv(nombre_csv, index=False)
    print(f"📁 CSV guardado en: {nombre_csv}")
    return df

def generar_graficas(df, detector_id, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_ram = os.path.join(carpeta_salida, f"grafica_RAM_{detector_id}.png")
    ruta_cpu = os.path.join(carpeta_salida, f"grafica_CPU_{detector_id}.png")

    # RAM
    plt.figure()
    plt.plot(df["tiempo_s"], df["RAM_MB"])
    plt.title(f"Uso de RAM - Detector {detector_id}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("RAM (MB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ruta_ram)

    # CPU
    plt.figure()
    plt.plot(df["tiempo_s"], df["CPU_acumulado_s"])
    plt.title(f"Tiempo de CPU acumulado - Detector {detector_id}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("CPU usado (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ruta_cpu)

    print(f"🖼️ Gráficas guardadas en: {carpeta_salida}")

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_detector = os.path.normpath(os.path.join(ruta_actual, RUTA_SCRIPT_DETECTOR))
    ruta_csv = os.path.join(ruta_actual, NOMBRE_CSV)
    ruta_graficas = os.path.join(ruta_actual, CARPETA_GRAFICAS)

    print(f"🚀 Ejecutando detector: {ruta_detector}")
    print(f"📊 Guardando CSV: {ruta_csv}")
    print(f"🖼️ Guardando gráficas en: {ruta_graficas}")

    if not os.path.exists(ruta_detector):
        print(f"❌ ERROR: No se encontró el script del detector en:\n{ruta_detector}")
        exit(1)

    proceso = lanzar_detector(ruta_detector)
    pid = proceso.pid
    print(f"🔍 Monitorizando PID: {pid}")

    df_resultado = monitor_recursos(ruta_csv, pid)
    proceso.wait()

    generar_graficas(df_resultado, DETECTOR_ID, ruta_graficas)
    print("✅ Finalizado correctamente.")