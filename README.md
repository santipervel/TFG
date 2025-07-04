# Diseño e implementación de un sistema de detección de anomalías en datos de usuarios

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Grado (TFG), centrado en la detección de comportamientos anómalos en logs de usuarios mediante técnicas estadísticas y de aprendizaje automático. Los scripts están organizados por detectores y se pueden ejecutar de forma independiente.

## 📁 Estructura del proyecto

```text
├── split.py                        # Script para dividir el dataset CLUE-LDS por periodos
├── DETECTORES/
│   ├── DetectorPerfilComportamiento/
│   │   ├── D1_anomaliaA1.py
│   │   ├── D1_anomaliaA3.py
│   │   ├── GraficasMetricas.py
│   │   ├── Mejores-Peores_Usuarios.py
│   │   └── SeleccionListaUsuarios.py
│   ├── DetectorPerfilHorario/
│   │   ├── D2.py
│   │   ├── GraficasMetricas.py
│   │   ├── GraficasUsuarios.py
│   │   ├── MejoresyPeoresUsuarios.py
│   │   └── SeleccionListaEventos.py
│   ├── DetectorDiaSemana/
│   │   ├── D3.py
│   │   ├── D4.py
│   │   ├── CalculoMedias.py
│   │   ├── GraficasMetricas.py
│   │   └── Mejores_Peores_Usuarios.py
│   └── DetectorEWMA/
│       ├── D5.py
│       ├── D6.py
│       ├── GraficasMetricas.py
│       └── GraficasUsuarios.py
```




## ⚙️ Flujo de ejecución
El proyecto sigue una estructura en tres fases: preparación de datos, ejecución del detector y análisis de resultados.

1️º Preparación previa
Antes de ejecutar los detectores es necesario generar las listas de usuarios o eventos filtrados:

- Ejecutar el script SeleccionListaUsuarios.py: este script crea un archivo csv con la lista de usuarios seleccionados que usaran los detectores

- Ejecutar el script SeleccionListaEventos.py: este script imprime la lista de eventos seleccionados (este no es necesario porque ya esta incluido en los detectores

2º Ejecución del detector
Cada detector tiene un script principal (D1_anomaliaA1.py, D2.py, etc.) que analiza el comportamiento y genera los resultados:

Los detectores crean archivos .csv con las métricas y resultados de anomalías detectadas.

3º Visualización y análisis
Una vez obtenidos los resultados para un detector en específico, se pueden lanzar scripts auxiliares para generar gráficas, comparar usuarios y visualizar métricas.

GraficasMetricas.py, Mejores-Peores_Usuarios.py, GraficasUsuarios.py y CalculoMedias.py: estos scripts leen los archivos .csv generados previamente y generan representaciones gráficas y o archivos csv con más resultados.

