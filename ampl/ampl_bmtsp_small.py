import os
import math
import time
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
from amplpy import AMPL, Environment

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent  # Directorio raíz del proyecto
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
        # Saltar líneas de encabezado hasta NODE_COORD_SECTION
        line = f.readline().strip()
        while line and 'NODE_COORD_SECTION' not in line:
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[-1].strip())
            line = f.readline().strip()
        
        # Leer coordenadas de las ciudades
        for line in f:
            line = line.strip()
            if line == 'EOF':
                break
            parts = line.split()
            if len(parts) >= 3:  # Asegurarse de tener al menos índice, x, y
                try:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append(City(idx-1, x, y))  # Ajustar índices a base 0
                except (ValueError, IndexError) as e:
                    print(f"  Warning: No se pudo analizar la línea: {line} - {e}")
                    continue
    return cities

def solve_bmtsp(cities: List[City], k: int, max_cities: int, problem_name: str = "") -> Dict[str, Any]:
    """Resolver el BMTSP usando el modelo existente."""
    start_time = time.time()
    
    # Configurar AMPL
    ampl = AMPL()
    ampl.set_option('solver', 'cplex')
    ampl.set_option('cplex_options', 'timelimit=600')  # 10 minutos máximo
    
    try:
        # Cargar el modelo
        print("  Cargando modelo...")
        ampl.read(str(MODEL_DIR / "modelo_bmtsp.mod"))
        
        # Configurar conjuntos y parámetros
        n = len(cities) - 1  # Excluyendo el depósito
        nodes = list(range(len(cities)))  # Índices de 0 a n
        
        # Crear matriz de distancias
        print("  Calculando distancias...")
        dist = {}
        for i in nodes:
            for j in nodes:
                if i != j:
                    dist[i, j] = cities[i].distance_to(cities[j])
        
        # Establecer parámetros
        print("  Configurando parámetros...")
        ampl.param['n'] = n
        ampl.param['k'] = k
        ampl.param['max_ciudades'] = max_cities
        
        # Establecer matriz de distancias
        ampl.param['dist'] = dist
        
        # Resolver
        print("  Resolviendo...")
        ampl.solve()
        
        # Obtener resultados
        x = ampl.get_variable('x')
        obj_value = ampl.get_objective('Distancia_Total').value()
        solve_time = time.time() - start_time
        
        # Extraer rutas
        routes = []
        route_lengths = []
        
        # Para cada vendedor
        for s in range(1, k + 1):
            route = [0]  # Empezar en el depósito (nodo 0)
            current = 0
            visited = set([0])
            
            # Construir ruta
            while True:
                next_node = None
                for j in nodes:
                    if j != current and x.get([s, current, j]).value() > 0.5:
                        next_node = j
                        break
                
                if next_node is None or next_node in visited:
                    break
                    
                route.append(next_node)
                visited.add(next_node)
                current = next_node
                
                # Si volvemos al depósito, terminar
                if current == 0:
                    break
            
            # Asegurar que la ruta termine en el depósito
            if len(route) > 1 and route[-1] != 0:
                route.append(0)
                
            # Calcular longitud de la ruta
            if len(route) > 1:  # Ruta válida
                length = sum(cities[route[i]].distance_to(cities[route[i+1]]) 
                           for i in range(len(route)-1))
                route_lengths.append(length)
                routes.append(route)
        
        # Calcular métricas
        metrics = {
            'problem': problem_name,
            'total_cost': obj_value,
            'routes': routes,
            'route_lengths': route_lengths,
            'num_routes': len(route_lengths),
            'execution_time': solve_time,
            'ampl_status': ampl.get_value('solve_result')
        }
        
        # Calcular estadísticas de rutas
        if route_lengths:
            metrics.update({
                'avg_route_length': statistics.mean(route_lengths),
                'min_route_length': min(route_lengths),
                'max_route_length': max(route_lengths),
                'std_dev_length': statistics.stdev(route_lengths) if len(route_lengths) > 1 else 0
            })
        else:
            metrics.update({
                'avg_route_length': 0,
                'min_route_length': 0,
                'max_route_length': 0,
                'std_dev_length': 0
            })
            
        return metrics
        
    except Exception as e:
        print(f"Error resolviendo con AMPL: {e}")
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
        f.write(f"SOLUCIÓN AMPL BMTSP - {results.get('problem', '')}\n")
        f.write("="*60 + "\n")
        
        if 'error' in results:
            f.write(f"Error: {results['error']}\n")
            return
        
        f.write(f"Costo total: {results.get('total_cost', 0):.2f}\n")
        f.write(f"Número de rutas: {results.get('num_routes', 0)}\n")
        f.write(f"Tiempo de ejecución: {results.get('execution_time', 0):.2f} segundos\n")
        f.write(f"Estado del solver: {results.get('ampl_status', 'N/A')}\n\n")
        
        f.write("ESTADÍSTICAS DE RUTAS:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Ruta':<8} | {'Ciudades':<8} | {'Longitud':<12} | Camino\n")
        f.write("-"*80 + "\n")
        
        for i, (route, length) in enumerate(zip(results.get('routes', []), results.get('route_lengths', [])), 1):
            num_cities = len(route) - 2  # Excluir depósito al inicio/fin
            f.write(f"{i:<8} | {num_cities:<8} | {length:>12.2f} | {' -> '.join(map(str, route))}\n")
        
        f.write("\nESTADÍSTICAS RESUMEN:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Longitud promedio:':<25} {results.get('avg_route_length', 0):.2f}\n")
        f.write(f"{'Ruta más corta:':<25} {results.get('min_route_length', 0):.2f}\n")
        f.write(f"{'Ruta más larga:':<25} {results.get('max_route_length', 0):.2f}\n")
        f.write(f"{'Desviación estándar:':<25} {results.get('std_dev_length', 0):.2f}\n")

def main():
    # Archivo TSP a procesar (instancia personalizada muy pequeña)
    tsp_file = 'mini10.tsp'  # Solo 10 nodos - instancia personalizada
    k = 1  # Solo 1 vendedor (TSP estándar)
    max_cities = 10  # Máximo de ciudades igual al total
    
    # Crear archivo de resultados
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f'bmtsp_{tsp_file.replace(".tsp", "")}_{timestamp}.txt'
    
    print("="*60)
    print(f"PROCESANDO: {tsp_file}")
    print("="*60)

    
    # Cargar ciudades
    print("  Cargando ciudades...")
    cities = load_tsp_file(str(DATA_DIR / tsp_file))
    print(f"  Se cargaron {len(cities)} ciudades")
    
    # Resolver
    print(f"  Resolviendo con AMPL (k={k}, max_ciudades={max_cities})...")
    results = solve_bmtsp(cities, k, max_cities, tsp_file)
    
    # Guardar resultados
    if 'error' not in results:
        save_results(results, str(output_file))
        print(f"\n¡Solución encontrada!")
        print(f"  Costo total: {results['total_cost']:.2f}")
        print(f"  Tiempo de ejecución: {results['execution_time']:.2f} segundos")
        print(f"  Resultados guardados en: {output_file}")
    else:
        print(f"\nError: {results['error']}")
    
    print("\n" + "="*60)
    print("EJECUCIÓN COMPLETADA")
    print("="*60)

if __name__ == "__main__":
    main()
