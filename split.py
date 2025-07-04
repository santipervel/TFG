import os

# Configuración
input_file = 'c:/Users/USUARIO/OneDrive/Escritorio/TFG/Codigo/clue.json'
output_prefix = 'c:/Users/USUARIO/OneDrive/Escritorio/TFG/Codigo/split_part_'
lines_per_file = 100000  # Número de líneas por cada fichero dividido

# Contador y escritura de ficheros divididos
current_part = 0
output_file = f'{output_prefix}{current_part}.jsonl'
out = open(output_file, 'w', encoding='utf-8')

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i > 0 and i % lines_per_file == 0:
            out.close()
            current_part += 1
            output_file = f'{output_prefix}{current_part}.jsonl'
            out = open(output_file, 'w', encoding='utf-8')

        out.write(line)

out.close()
print(f'Split completado. Archivo dividido en {current_part + 1} partes.')
