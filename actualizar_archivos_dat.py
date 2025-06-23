import os
import re

def estandarizar_archivo(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        contenido = f.read()
    
    # Reemplazar max_ciudades_por_vendedor por max_ciudades
    contenido = re.sub(r'param max_ciudades_por_vendedor', 'param max_ciudades', contenido)
    
    # Reemplazar d : por dist :
    contenido = re.sub(r'param d\s*:', 'param dist :', contenido)
    
    # Procesar la matriz de distancias para asegurar que los índices sean enteros
    # y redondear los valores de distancia
    lines = contenido.split('\n')
    dist_start_idx = -1
    dist_end_idx = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith('param dist :'):
            dist_start_idx = i
        elif dist_start_idx != -1 and ';' in line:
            dist_end_idx = i
            break
    
    if dist_start_idx != -1 and dist_end_idx != -1:
        # Procesar las líneas de la matriz de distancias
        for i in range(dist_start_idx + 1, dist_end_idx):
            parts = lines[i].split()
            if parts:
                # Asegurar que el primer elemento sea un entero (índice de ciudad)
                try:
                    idx = int(float(parts[0]))
                    parts[0] = str(idx)
                    # Redondear las distancias a enteros
                    for j in range(1, len(parts)):
                        try:
                            val = float(parts[j])
                            parts[j] = str(int(round(val)))
                        except ValueError:
                            pass
                    lines[i] = ' '.join(parts)
                except ValueError:
                    pass
    
    contenido = '\n'.join(lines)
    
    # Escribir el archivo actualizado
    with open(ruta_archivo, 'w') as f:
        f.write(contenido)

def main():
    # Directorio que contiene los archivos .dat
    directorio = os.path.join('ampl', 'datos')
    
    # Lista de archivos a actualizar (excluyendo datos_bmtsp.dat que es nuestra referencia)
    archivos = [
        'pequeno_10ciudades_2vendedores.dat',
        'mediano_20ciudades_3vendedores.dat',
        'grande_30ciudades_4vendedores.dat',
        'muy_grande_50ciudades_5vendedores.dat',
        'desafio_100ciudades_8vendedores.dat'
    ]
    
    for archivo in archivos:
        ruta = os.path.join(directorio, archivo)
        if os.path.exists(ruta):
            print(f"Actualizando {archivo}...")
            estandarizar_archivo(ruta)
            print(f"{archivo} actualizado correctamente.")
        else:
            print(f"Advertencia: No se encontró el archivo {ruta}")

if __name__ == "__main__":
    main()
