import os
import math
import time
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
from amplpy import AMPL, Environment

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent  # Subir un nivel para llegar a la raíz del proyecto
MODEL_DIR = BASE_DIR / "ampl"  # Directorio del modelo
DATA_DIR = BASE_DIR / "LKH3.0" / "instances"  # Directorio de datos TSP
RESULTS_DIR = BASE_DIR / "ampl_results"  # Directorio de resultados
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

@dataclass
class City:
    idx: int
    x: float
    y: float

    def distance_to(self, other: "City") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

def load_tsp_file(file_path: str) -> List[City]:
    """Cargar ciudades desde un archivo TSPLIB."""
    cities = []
    with open(file_path, 'r') as f:
        # Skip header lines until NODE_COORD_SECTION
        line = f.readline().strip()
        while line and 'NODE_COORD_SECTION' not in line:
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[-1].strip())
            line = f.readline().strip()
        
        # Read city coordinates
        for line in f:
            line = line.strip()
            if line == 'EOF':
                break
            parts = line.split()
            if len(parts) >= 3:  # Ensure we have at least index, x, y
                try:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append(City(idx-1, x, y))  # Ajustar índices a base 0
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not parse line: {line} - {e}")
                    continue
    return cities

def solve_bmtsp(cities: List[City], k: int, max_cities: int, problem_name: str = "") -> Dict[str, Any]:
    """Resolver el BMTSP usando el modelo existente."""
    start_time = time.time()
    
    # Configurar AMPL
    ampl = AMPL()
    ampl.set_option('solver', 'cplex')
    ampl.set_option('cplex_options', 'timelimit=3600')
    
    try:
        # Cargar el modelo
        print("  Loading model...")
        ampl.read(str(MODEL_DIR / "modelo_bmtsp.mod"))
        
        # Configurar conjuntos y parámetros
        n = len(cities) - 1  # Excluyendo el depósito
        nodes = list(range(len(cities)))  # Índices de 0 a n
        
        # Crear matriz de distancias
        print("  Calculating distances...")
        dist = {}
        for i in nodes:
            for j in nodes:
                if i != j:
                    dist[i, j] = cities[i].distance_to(cities[j])
        
        # Establecer parámetros
        print("  Setting parameters...")
        ampl.param['n'] = n
        ampl.param['k'] = k
        ampl.param['max_ciudades'] = max_cities
        
        # Establecer matriz de distancias
        ampl.param['dist'] = dist
        
        # Resolver
        print("  Solving...")
        ampl.solve()
        
        # Obtener resultados
        solution = ampl.get_variable('x').get_values()
        obj_value = ampl.get_objective('Distancia_Total').value()
        solve_time = time.time() - start_time
        
        # Procesar rutas
        routes = extract_routes(solution, nodes, k)
        
        # Calcular métricas
        route_lengths = []
        for route in routes:
            if len(route) > 1:  # Ruta válida
                length = sum(cities[u].distance_to(cities[v]) for u, v in zip(route, route[1:]))
                route_lengths.append(length)
        
        return {
            'problem': problem_name,
            'total_cost': obj_value,
            'routes': routes,
            'route_lengths': route_lengths,
            'num_routes': len(route_lengths),
            'avg_route_length': statistics.mean(route_lengths) if route_lengths else 0,
            'min_route_length': min(route_lengths) if route_lengths else 0,
            'max_route_length': max(route_lengths) if route_lengths else 0,
            'std_dev_length': statistics.stdev(route_lengths) if len(route_lengths) > 1 else 0,
            'execution_time': solve_time,
            'ampl_status': ampl.get_value('solve_result')
        }
        
    except Exception as e:
        print(f"Error solving with AMPL: {e}")
        return {'error': str(e)}
    finally:
        ampl.close()

def extract_routes(solution: Dict, nodes: List[int], k: int) -> List[List[int]]:
    """Extraer rutas de la solución de AMPL."""
    routes = []
    
    # Para cada vendedor
    for s in range(1, k + 1):
        route = [0]  # Comenzar en el depósito
        current = 0
        visited = set()
        
        # Construir ruta
        while True:
            next_node = None
            # Buscar siguiente nodo
            for j in nodes:
                if j != current and solution.get((s, current, j), 0) > 0.9:
                    next_node = j
                    break
            
            if next_node is None or next_node in visited:
                break
                
            route.append(next_node)
            visited.add(next_node)
            current = next_node
            
            # Si volvemos al depósito, terminar la ruta
            if current == 0:
                break
        
        # Asegurarse de que la ruta termina en el depósito
        if len(route) > 1 and route[-1] != 0:
            route.append(0)
            
        routes.append(route)
    
    return routes

def save_results(results: Dict, output_file: str):
    """Guardar los resultados en un archivo."""
    with open(output_file, 'w') as f:
        f.write(f"AMPL BMTSP SOLUTION - {results.get('problem', '')}\n")
        f.write("="*60 + "\n")
        
        if 'error' in results:
            f.write(f"Error: {results['error']}\n")
            return
        
        f.write(f"Total cost: {results.get('total_cost', 0):.2f}\n")
        f.write(f"Number of routes: {results.get('num_routes', 0)}\n")
        f.write(f"Execution time: {results.get('execution_time', 0):.2f} seconds\n")
        f.write(f"Solver status: {results.get('ampl_status', 'N/A')}\n\n")
        
        f.write("ROUTE STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Route':<8} | {'Cities':<8} | {'Length':<12} | Path\n")
        f.write("-"*80 + "\n")
        
        for i, (route, length) in enumerate(zip(results.get('routes', []), results.get('route_lengths', [])), 1):
            num_cities = len(route) - 2  # Excluir depósito al inicio/fin
            f.write(f"{i:<8} | {num_cities:<8} | {length:>12.2f} | {' -> '.join(map(str, route))}\n")
        
        f.write("\nSUMMARY STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Average route length:':<25} {results.get('avg_route_length', 0):.2f}\n")
        f.write(f"{'Shortest route:':<25} {results.get('min_route_length', 0):.2f}\n")
        f.write(f"{'Longest route:':<25} {results.get('max_route_length', 0):.2f}\n")
        f.write(f"{'Standard deviation:':<25} {results.get('std_dev_length', 0):.2f}\n")

def main():
    # Archivos TSP a procesar
    tsp_files = [
        'kroA100.tsp',
        'kroB100.tsp',
        'kroC100.tsp',
        'kroD100.tsp',
        'kroE100.tsp'
    ]
    
    k = 5  # Número de vendedores
    max_cities = 20  # Máximo de ciudades por vendedor
    
    # Crear archivo de resumen
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_file = RESULTS_DIR / f'bmtsp_summary_{timestamp}.txt'
    
    with open(summary_file, 'w') as sf:
        sf.write("AMPL BMTSP SOLVER - COMPARATIVE RESULTS\n")
        sf.write("="*60 + "\n")
        sf.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        sf.write(f"Number of salesmen: {k}\n")
        sf.write(f"Max cities per salesman: {max_cities}\n\n")
        sf.write(f"{'PROBLEM':<15} | {'TOTAL COST':>12} | {'BEST ROUTE':>12} | "
                f"{'WORST ROUTE':>12} | {'STD DEV':>8} | {'TIME (s)':>8} | {'STATUS'}\n")
        sf.write("-"*100 + "\n")
    
    # Procesar cada archivo TSP
    for tsp_file in tsp_files:
        file_path = DATA_DIR / tsp_file
        problem_name = file_path.stem
        
        if not file_path.exists():
            print(f"File {tsp_file} not found, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"PROCESSING: {tsp_file}")
        print(f"{'='*60}")
        
        try:
            # Cargar ciudades
            print("  Loading cities...")
            cities = load_tsp_file(str(file_path))
            if not cities:
                print(f"  Error: No cities loaded from {tsp_file}")
                continue
                
            print(f"  Loaded {len(cities)} cities")
            
            # Resolver con AMPL
            print(f"  Solving with AMPL (k={k}, max_cities={max_cities})...")
            results = solve_bmtsp(cities, k, max_cities, problem_name)
            
            if 'error' in results:
                print(f"  Error: {results['error']}")
                status = "ERROR"
            else:
                status = results.get('ampl_status', 'UNKNOWN')
                print(f"  Solution found with total cost: {results['total_cost']:.2f}")
                print(f"  Execution time: {results['execution_time']:.2f} seconds")
                
                # Guardar resultados detallados
                output_file = RESULTS_DIR / f"{problem_name}_bmtsp_result.txt"
                save_results(results, str(output_file))
                print(f"  Detailed results saved to: {output_file}")
            
            # Actualizar resumen
            with open(summary_file, 'a') as sf:
                sf.write(
                    f"{problem_name:<15} | "
                    f"{results.get('total_cost', 0):>12.2f} | "
                    f"{results.get('min_route_length', 0):>12.2f} | "
                    f"{results.get('max_route_length', 0):>12.2f} | "
                    f"{results.get('std_dev_length', 0):>8.2f} | "
                    f"{results.get('execution_time', 0):>8.2f} | "
                    f"{status}\n"
                )
                
        except Exception as e:
            print(f"  Error processing {tsp_file}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*60)
    print("EXECUTION COMPLETED")
    print("="*60)
    print(f"Summary file: {os.path.abspath(summary_file)}")

if __name__ == "__main__":
    main()