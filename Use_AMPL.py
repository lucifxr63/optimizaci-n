import os
import subprocess
import math
import csv
import matplotlib.pyplot as plt

class Ciudad:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = float(x)
        self.y = float(y)

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

def load_cities(csv_path):
    cities = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # saltar encabezado
        for row in reader:
            if len(row) >= 3:
                cities.append(Ciudad(int(row[0]), row[1], row[2]))
    return cities

def reconstruir_ruta(conexiones):
    if not conexiones:
        return [0]
    inicio = 0 if 0 in conexiones else next(iter(conexiones))
    ruta = [inicio]
    actual = inicio
    visitados = set()
    while actual in conexiones and actual not in visitados:
        visitados.add(actual)
        siguiente = conexiones[actual]
        ruta.append(siguiente)
        actual = siguiente
        if siguiente == ruta[0]:
            break
    if 0 in ruta and ruta[0] != 0:
        idx = ruta.index(0)
        ruta = ruta[idx:] + ruta[1:idx+1]
    if ruta[-1] != 0:
        ruta.append(0)
    return ruta

def procesar_rutas(ampl_output):
    import re
    conexiones = {}
    lineas = ampl_output.split('\n')
    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()
        match_vend = re.match(r"(x\s*)?\[\s*(\d+),\*,\*\]", linea)
        if match_vend:
            vendedor = int(match_vend.group(2))
            conexiones[vendedor] = {}
            while i < len(lineas) and ":=" not in lineas[i]:
                i += 1
            i += 1
            while i < len(lineas) and lineas[i].strip() and not lineas[i].strip().startswith('['):
                partes = lineas[i].strip().split()
                if len(partes) > 1:
                    origen = int(partes[0])
                    for j, val in enumerate(partes[1:]):
                        try:
                            if float(val) > 0.5:
                                conexiones[vendedor][origen] = j
                        except ValueError:
                            pass
                i += 1
        else:
            i += 1
    rutas = {}
    for vendedor, links in conexiones.items():
        rutas[vendedor] = reconstruir_ruta(links)
    return rutas

def plot_routes_per_salesman(rutas, ciudades):
    colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    idx_to_city = {c.idx: c for c in ciudades}

    plt.figure(figsize=(10, 6))
    for v, ruta in rutas.items():
        color = colores[(v - 1) % len(colores)]
        x_coords = [idx_to_city[i].x for i in ruta]
        y_coords = [idx_to_city[i].y for i in ruta]
        plt.plot(x_coords, y_coords, marker='o', label=f"Vendedor {v}", color=color)
        for i, ci in enumerate(ruta):
            plt.text(idx_to_city[ci].x + 0.2, idx_to_city[ci].y + 0.2, str(ci), fontsize=9)

    # Destacar el depósito (ciudad 0)
    if 0 in idx_to_city:
        depot = idx_to_city[0]
        plt.scatter([depot.x], [depot.y], s=100, c='black', marker='s', label='Depósito')

    plt.title("Rutas por Vendedor")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_ampl():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ampl_path = r"D:\DEV\AMPL\ampl.exe"
    model_path = os.path.join(base_dir, "ampl", "modelo_bmtsp.mod")
    data_path = os.path.join(base_dir, "ampl", "datos_bmtsp.dat")
    csv_path = os.path.join(base_dir, "python", "ciudades.csv")  # ajusta si es necesario

    ciudades = load_cities(csv_path)
    n = len(ciudades) - 1
    k = 2
    max_ciudades = 5

    with open(data_path, 'w') as f:
        f.write(f"param n := {n};\n")
        f.write(f"param k := {k};\n")
        f.write(f"param max_ciudades := {max_ciudades};\n")
        f.write("param dist : " + " ".join(str(c.idx) for c in ciudades) + " :=\n")
        for ci in ciudades:
            f.write(str(ci.idx) + " " + " ".join(str(int(ci.distance_to(cj))) for cj in ciudades) + "\n")
        f.write(";\n")

    cmd = f"""
    model \"{model_path}\"; 
    data \"{data_path}\"; 
    option solver cplex; 
    solve; 
    display n, k, max_ciudades; 
    display Total_Distance; 
    display x;
    """

    print("Ejecutando AMPL con el comando:")
    print(cmd)

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run') as f:
            f.write(cmd)
            temp_file = f.name

        result = subprocess.run([ampl_path, temp_file], text=True, capture_output=True)

        print("\n=== SALIDA DE AMPL ===")
        print(result.stdout)

        rutas = procesar_rutas(result.stdout)

                # Extraer costo total desde la salida
        distancia_total = None
        for line in result.stdout.splitlines():
            if "Total_Distance" in line and "=" in line:
                try:
                    distancia_total = float(line.split('=')[1].strip())
                except:
                    pass

        print("\n=== RUTAS ASIGNADAS ===")
        for v in range(1, k + 1):
            if v in rutas:
                ruta = rutas[v]
                print(f"Vendedor {v}: {' → '.join(str(x) for x in ruta)}")
            else:
                print(f"Vendedor {v}: No tiene ruta asignada")

        if distancia_total is not None:
            print(f"\nCosto total (distancia recorrida): {distancia_total:.2f} km")
        else:
            print("\nNo se pudo obtener el costo total desde la salida de AMPL.")


        # Mostrar mapa con rutas
        plot_routes_per_salesman(rutas, ciudades)

    except Exception as e:
        print(f"Error al ejecutar AMPL: {e}")

if __name__ == "__main__":
    test_ampl()
