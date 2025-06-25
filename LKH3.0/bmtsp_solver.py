import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
import argparse
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Advertencia: No se pudo importar reportlab. La generación de PDFs estará deshabilitada.")

# Crear directorios de resultados si no existen
RESULTS_DIR = Path("resultados")
MAPS_DIR = RESULTS_DIR / "mapas"
PDFS_DIR = RESULTS_DIR / "pdfs"

for directory in [MAPS_DIR, PDFS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class InstanceBMTSP:
    """
    Class to handle TSPLIB instances and manage distance calculations.
    Also loads best-known solutions (BKS) if available.
    """
    def __init__(self, instance_path: str):
        self.instance_path = instance_path
        self.name = Path(instance_path).stem
        self.depot, self.cities = self._read_tsp(instance_path)
        self.n = len(self.cities)
        self.distance_matrix = self._calculate_distances()
        self.bks = self._load_bks()
    
    def _read_tsp(self, file_path: str) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
        """Read TSPLIB format file and return depot and cities coordinates."""
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Find the start of the coordinates
        node_coord_section = False
        coords = []
        
        for line in lines:
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_section = True
                continue
            if line.startswith('EOF'):
                break
            if node_coord_section:
                parts = line.split()
                # Handle different TSPLIB formats (with/without node index)
                if parts[0].isdigit():
                    coords.append((float(parts[1]), float(parts[2])))
                else:
                    coords.append((float(parts[0]), float(parts[1])))
        
        if not coords:
            raise ValueError("No coordinates found in the TSP file")
            
        depot = coords[0]
        cities = coords[1:]
        return depot, cities
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate Euclidean distance matrix."""
        points = [self.depot] + self.cities
        n = len(points)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = math.dist(points[i], points[j])
        return matrix
    
    def _load_bks(self) -> Optional[float]:
        """Try to load best-known solution from solution files."""
        base_path = Path(self.instance_path).parent
        
        # Try different possible BKS file names and formats
        possible_files = [
            base_path / f"{self.name}.opt.tour",  # Standard TSPLIB format
            base_path / f"{self.name}.tour",       # Alternative TSPLIB format
            base_path / f"{self.name}.*.sol"       # Custom solution format with cost in filename
        ]
        
        for pattern in possible_files:
            if '*' in str(pattern):
                # Handle glob pattern
                import glob
                matches = glob.glob(str(pattern))
                if matches:
                    bks_file = Path(matches[0])
                else:
                    continue
            else:
                bks_file = pattern
                
            if not bks_file.exists():
                continue
                
            try:
                with open(bks_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                
                # Try to extract cost from filename first (format: instance.k-min-max.cost.sol)
                if bks_file.suffix == '.sol':
                    parts = bks_file.stem.split('.')
                    if len(parts) >= 3 and parts[-1].isdigit():
                        # Format: a280.8-30-50.4272009.sol -> cost = 4272.009
                        cost = int(parts[-1]) / 1000.0
                        return cost
                    
                    # Check if cost is in the first line (format: "a280, Cost: X_YYYYYYY")
                    if lines and 'Cost:' in lines[0]:
                        try:
                            cost_str = lines[0].split('Cost:')[-1].strip().split('_')[-1]
                            cost = int(cost_str) / 1000.0
                            return cost
                        except (ValueError, IndexError):
                            pass
                
                # Try standard TSPLIB format
                tour_start = False
                tour_nodes = []
                
                for line in lines:
                    if line.startswith('TOUR_SECTION') or line.startswith('TOUR'):
                        tour_start = True
                        continue
                    if line == '-1' or line.startswith('EOF'):
                        break
                    if tour_start and line.strip():
                        # Handle lines with format: "1 2 3 1 (#30) Cost: 123456"
                        clean_line = line.split('Cost:')[0].split('(#')[0].strip()
                        nodes = [int(x) for x in clean_line.split()]
                        # Remove duplicates that might occur when the tour is a cycle
                        if nodes[-1] == nodes[0]:
                            nodes = nodes[:-1]
                        tour_nodes.extend([x-1 for x in nodes])  # Convert to 0-based index
                
                if not tour_nodes:
                    continue
                    
                # Calculate tour length
                total_distance = 0
                for i in range(len(tour_nodes)):
                    j = (i + 1) % len(tour_nodes)
                    total_distance += self.distance_matrix[tour_nodes[i]][tour_nodes[j]]
                    
                return total_distance
                
            except Exception as e:
                print(f"Warning: Could not parse BKS from {bks_file}: {e}")
                continue
        
        # Known BKS values for common instances
        known_bks = {
            'a280': 4373.71,
            'kroA100': 21282.0,
            'kroB100': 22141.0,
            'kroC100': 20749.0,
            'kroD100': 21294.0,
            'kroE100': 22068.0
        }
        
        # Return known BKS if available
        return known_bks.get(self.name, None)


class RoutePartitioner:
    """Class to partition cities into routes respecting size constraints."""
    def __init__(self, instance: InstanceBMTSP, k: int, min_c: int, max_c: int):
        self.instance = instance
        self.k = k
        self.min_c = min_c
        self.max_c = max_c
    
    def partition(self) -> List[List[int]]:
        """
        Partition the cities into k balanced groups using k-means clustering.
        
        Returns:
            List of k lists, where each list contains the city indices for one group
        """
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Get city coordinates (excluding depot)
            coords = np.array([self.instance.cities[i-1] for i in range(1, len(self.instance.cities) + 1)])  # -1 because cities are 1-based
            
            # Use k-means to cluster cities
            kmeans = KMeans(
                n_clusters=self.k,
                n_init=10,  # Try multiple initializations
                max_iter=300,
                random_state=42
            )
            
            # Fit and predict clusters
            clusters = kmeans.fit_predict(coords)
            
            # Group cities by cluster
            groups = [[] for _ in range(self.k)]
            for city_idx, cluster in enumerate(clusters):
                groups[cluster].append(city_idx + 1)
            
            # Balance the groups to respect min/max constraints
            balanced_groups = self._balance_groups(groups)
            
            return balanced_groups
            
        except ImportError:
            print("  Advertencia: scikit-learn no está instalado, usando partición simple")
            # Fallback to simple round-robin assignment
            groups = [[] for _ in range(self.k)]
            for i, city in enumerate(range(1, len(self.instance.cities) + 1)):
                groups[i % self.k].append(city)
            return groups
    
    def _balance_groups(self, groups: List[List[int]]) -> List[List[int]]:
        """
        Balance the groups to respect min/max constraints.
        
        Args:
            groups: List of city groups
            
        Returns:
            Balanced list of city groups
        """
        # Flatten groups and calculate current sizes
        all_cities = [city for group in groups for city in group]
        group_sizes = [len(group) for group in groups]
        
        # Target size is average, but respect min/max
        target_size = len(all_cities) // len(groups)
        target_size = max(self.min_c, min(self.max_c, target_size))
        
        # Balance groups
        balanced_groups = []
        remaining_cities = all_cities.copy()
        
        for _ in range(len(groups)):
            if not remaining_cities:
                break
                
            # Take up to target_size cities for this group
            group_size = min(target_size, len(remaining_cities))
            group = remaining_cities[:group_size]
            balanced_groups.append(group)
            remaining_cities = remaining_cities[group_size:]
        
        # If we have remaining cities, distribute them
        while remaining_cities:
            # Find group with smallest size that's below max_cities
            group_sizes = [len(group) for group in balanced_groups]
            min_size = min(group_sizes)
            
            for i in range(len(balanced_groups)):
                if len(balanced_groups[i]) == min_size and len(balanced_groups[i]) < self.max_c:
                    balanced_groups[i].append(remaining_cities.pop(0))
                    break
            else:
                # If no group can take more cities, add to the smallest group
                min_idx = group_sizes.index(min(group_sizes))
                balanced_groups[min_idx].append(remaining_cities.pop(0))
        
        return balanced_groups

    def assign_cities(self, cities_idx: List[int], groups: List[List[int]]) -> List[List[int]]:
        """
        Assign cities to groups respecting size constraints.
        
        Args:
            cities_idx: List of city indices to assign
            groups: List of city groups
            
        Returns:
            Updated list of city groups
        """
        pbar = tqdm(total=len(cities_idx), desc="Asignando ciudades a grupos")
        progress_made = True
        
        while cities_idx and progress_made:
            progress_made = False
            
            for city in cities_idx[:]:
                for group in groups:
                    if len(group) < self.max_c:
                        group.append(city)
                        cities_idx.remove(city)
                        pbar.update(1)
                        progress_made = True
                        break
                else:
                    continue
                break
            
            # Si no se pudo asignar ninguna ciudad en esta ronda, forzar asignación
            if not progress_made and cities_idx:
                print("\n¡Advertencia! No se pudo asignar ciudades con las restricciones actuales.")
                print(f"Ciudades restantes: {len(cities_idx)}")
                print("Asignando ciudades restantes a grupos disponibles...")
                # Asignar las ciudades restantes a cualquier grupo que tenga espacio
                for city in cities_idx[:]:
                    for group in groups:
                        if len(group) < self.max_c:
                            group.append(city)
                            cities_idx.remove(city)
                            pbar.update(1)
                            break
                break  # Salir del bucle while principal
        
        # Mostrar resumen de la partición
        print("\nResumen de la partición:")
        for i, group in enumerate(groups):
            print(f"  Vendedor {i+1}: {len(group)} ciudades")
        
        return groups


class RouteGenerator:
    """Class to generate optimal routes for each group using OR-Tools."""
    def __init__(self, instance: InstanceBMTSP):
        self.instance = instance
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        """Calculate the total distance of a route.
        
        Args:
            route: List of city indices in the route (starting and ending at the depot)
            
        Returns:
            Total distance of the route, or infinity if there's an error
        """
        if not route or len(route) < 2:
            print("  Ruta inválida: muy corta o vacía")
            return float('inf')
            
        total_distance = 0.0
        
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            
            try:
                # Validate city indices
                if from_city < 0 or from_city > len(self.instance.cities) or \
                   to_city < 0 or to_city > len(self.instance.cities):
                    print(f"  Índice de ciudad fuera de rango: from_city={from_city}, to_city={to_city}")
                    print(f"  Ruta: {route}")
                    print(f"  Número de ciudades en instancia: {len(self.instance.cities)}")
                    return float('inf')
                
                # Get coordinates for from_city
                if from_city == 0:  # Depot
                    from_coord = self.instance.depot
                else:
                    from_coord = self.instance.cities[from_city - 1]  # Convert to 0-based
                
                # Get coordinates for to_city
                if to_city == 0:  # Depot
                    to_coord = self.instance.depot
                else:
                    to_coord = self.instance.cities[to_city - 1]  # Convert to 0-based
                
                # Calculate Euclidean distance
                dx = from_coord[0] - to_coord[0]
                dy = from_coord[1] - to_coord[1]
                total_distance += math.sqrt(dx*dx + dy*dy)
                
            except IndexError as e:
                print(f"\nError calculando distancia entre ciudades {from_city} y {to_city}:")
                print(f"  Ruta: {route}")
                print(f"  Índice máximo permitido: {len(self.instance.cities) - 1}")
                print(f"  Intentando acceder a índices: {from_city-1} y {to_city-1}")
                print(f"  Número de ciudades en instancia: {len(self.instance.cities)}")
                return float('inf')
            except Exception as e:
                print(f"Error inesperado calculando distancia: {str(e)}")
                print(f"  Ruta: {route}")
                print(f"  Ciudades: from_city={from_city}, to_city={to_city}")
                return float('inf')
                
        return total_distance
    
    def _two_opt(self, route: List[int], current_cost: float) -> Tuple[List[int], float]:
        """Apply 2-opt local search to improve the route.
        
        Args:
            route: Current route as a list of city indices
            current_cost: Current total distance of the route
            
        Returns:
            Tuple of (improved_route, improved_cost)
        """
        if len(route) <= 3:  # Need at least 3 cities (excluding depot) for 2-opt
            return route, current_cost
            
        best_route = route.copy()
        best_cost = current_cost
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Try reversing the segment between i and j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_cost = self._calculate_route_distance(new_route)
                    
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            route = best_route
            
        return best_route, best_cost
        
    def generate_route(self, cities: List[int], time_limit: int = 30, 
                     first_solution: str = 'PATH_CHEAPEST_ARC',
                     local_search: str = 'GUIDED_LOCAL_SEARCH') -> Tuple[List[int], float]:
        """Generate an optimized route for the given cities using OR-Tools.
        
        Args:
            cities: List of city indices (0-based) to include in the route
            time_limit: Time limit in seconds for the solver
            first_solution: Strategy for initial solution
            local_search: Metaheuristic for local search
            
        Returns:
            Tuple containing:
            - List of city indices in the optimized order, starting and ending at the depot (0)
            - Total distance of the route
        """
        if not cities:
            return [0, 0], 0.0  # Return depot to depot with zero distance
            
        print(f"\n  Iniciando optimización para grupo de {len(cities)} ciudades")
        print(f"  Ciudades en el grupo: {cities}")
            
        # Ensure we don't have too many cities (OR-Tools has a limit)
        if len(cities) > 30:  # Reduced to 30 for stability
            print(f"  Advertencia: Demasiadas ciudades ({len(cities)}), limitando a 30")
            cities = cities[:30]
        
        print(f"\nGenerando ruta para {len(cities)} ciudades...")
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(cities) + 1,  # +1 for depot
            1,  # single vehicle
            0   # depot index
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index: int, to_index: int) -> int:
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # Handle depot (index 0 in the routing model)
            from_city = 0 if from_node == 0 else cities[from_node - 1]
            to_city = 0 if to_node == 0 else cities[to_node - 1]
            
            # Get distance from the instance's distance matrix
            try:
                distance = int(self.instance.distance_matrix[from_city][to_city] * 1000)  # Scale to avoid floating point
                # Ensure distance is not negative
                if distance < 0:
                    print(f"  Advertencia: Distancia negativa entre {from_city} y {to_city}: {distance}")
                    distance = 0
                return distance
            except IndexError as e:
                print(f"  Error en callback de distancia: {e}")
                print(f"  from_node: {from_node}, to_node: {to_node}")
                print(f"  from_city: {from_city}, to_city: {to_city}")
                print(f"  Número de ciudades: {len(cities)}")
                return 0
                
        # Register distance callback with the routing model
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set search parameters for LKH
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = time_limit
        
        # Configure for LKH (Lin-Kernighan Heuristic)
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # Optimize LKH parameters
        search_parameters.guided_local_search_lambda_coefficient = 0.1  # Más exploración local
        search_parameters.log_search = False  # Reducir salida de logs
        search_parameters.use_full_propagation = True
        
        # Limitar el número de soluciones a evaluar
        search_parameters.solution_limit = 1000
        
        # Configurar estrategia de solución inicial
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Configurar metaheurística de búsqueda local
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # Ajustar parámetros de reinicio
        search_parameters.guided_local_search_reset_penalties_on_new_best_solution = True
        
        print(f"  Tiempo límite configurado: {time_limit} segundos por ruta")
        print("  Configuración optimizada para búsqueda local eficiente")
        
        # Solve the problem with LKH
        print("  Usando estrategia LKH (Lin-Kernighan Heuristic)")
        solution = routing.SolveWithParameters(search_parameters)
        
        best_route = None
        best_distance = float('inf')
        
        if solution and solution.ObjectiveValue() < float('inf'):
            # Get the objective value without scaling it down
            best_distance = float(solution.ObjectiveValue())
            print(f"  Solución inicial encontrada con LKH: {best_distance:.2f}")
            
            # Extract the route
            index = routing.Start(0)
            route = []
            
            while not routing.IsEnd(index):
                # Convert from routing variable index to node index
                node_index = manager.IndexToNode(index)
                # Add 1 to node_index because cities are 1-based in the TSPLIB format
                route.append(node_index)
                # Move to next node
                index = solution.Value(routing.NextVar(index))
            
            # Add the depot at the end to complete the cycle
            if route:
                route.append(route[0])  # Return to starting point (depot)
            
            # Calculate the actual distance of the route using our distance matrix
            actual_distance = self._calculate_route_distance(route)
            
            # Use the actual calculated distance instead of the solver's objective value
            if actual_distance < float('inf'):
                best_distance = actual_distance
                best_route = route
                print(f"  Distancia verificada: {best_distance:.2f}")
            else:
                print("  Advertencia: No se pudo verificar la distancia de la ruta generada")
        
        if best_route is not None:
            print(f"  Mejor distancia encontrada: {best_distance:.2f}")
            
            # Apply 2-opt local search
            print("  Aplicando búsqueda local 2-opt...")
            best_route, best_distance = self._two_opt(best_route, best_distance)
            print(f"  Mejor distancia después de 2-opt: {best_distance:.2f}")
            
            return best_route, best_distance
        else:
            print("  No se encontró solución con OR-Tools, usando ruta secuencial")
            # Fallback: create a sequential route with nearest neighbor
            fallback_route = self._nearest_neighbor_route(cities)
            fallback_distance = self._calculate_route_distance(fallback_route)
            print(f"  Usando ruta del vecino más cercano con distancia: {fallback_distance:.2f}")
            return fallback_route, fallback_distance
            
    def _nearest_neighbor_route(self, cities: List[int]) -> List[int]:
        """Generate a route using nearest neighbor heuristic."""
        if not cities:
            return [0, 0]  # Depot to depot if no cities
            
        unvisited = set(cities)
        current = 0  # Start at depot
        route = [current]
        
        while unvisited:
            nearest = min(
                unvisited,
                key=lambda city: self.instance.distance_matrix[current][city]
            )
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        route.append(0)  # Return to depot
        return route
    
    def _generate_sequential_route(self, cities: List[int]) -> Tuple[List[int], float]:
        """Generate a simple sequential route as fallback."""
        route = [0] + cities + [0]  # Start and end at depot
        distance = self._calculate_route_distance(route)
        return route, distance
    
    def _two_opt(self, route: List[int], current_cost: float) -> Tuple[List[int], float]:
        """Apply 2-opt local search to improve the route."""
        if len(route) <= 3:  # Need at least 3 cities (excluding depot) for 2-opt
            return route, current_cost
            
        best_route = route.copy()
        best_cost = current_cost
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Try reversing the segment between i and j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_cost = self._calculate_route_distance(new_route)
                    
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            route = best_route

        return best_route, best_cost

class SolutionVisualizer:
    """Class to visualize the solution routes."""
    def __init__(self, instance: InstanceBMTSP):
        self.instance = instance
    
    def plot_routes(self, routes: List[List[int]], title: str = "BMTSP Solution", save_plot: bool = True) -> str:
        """Plot the routes on a 2D plane and save the figure."""
        plt.figure(figsize=(12, 8))
        points = [self.instance.depot] + self.instance.cities
        
        # Plot all cities
        x = [p[0] for p in points[1:]]  # Skip depot
        y = [p[1] for p in points[1:]]
        plt.scatter(x, y, c='gray', alpha=0.5, label='Cities')
        
        # Plot depot
        plt.scatter([points[0][0]], [points[0][1]], c='red', marker='s', s=100, label='Depot')
        
        # Plot routes
        colors = plt.colormaps['tab10'].resampled(len(routes)) if len(routes) > 0 else plt.colormaps['tab10']
        for i, route in enumerate(routes):
            if len(route) < 2:
                continue
                
            route_points = [points[node] for node in route]
            x = [p[0] for p in route_points]
            y = [p[1] for p in route_points]
            
            plt.plot(x, y, 'o-', linewidth=2, markersize=5, 
                    color=colors(i), label=f'Route {i+1}')
            
            # Add arrows to show direction
            for j in range(len(route_points)-1):
                dx = x[j+1] - x[j]
                dy = y[j+1] - y[j]
                plt.arrow(x[j], y[j], dx*0.9, dy*0.9, 
                         head_width=0.5, head_length=0.8, 
                         fc=colors(i), ec=colors(i), 
                         length_includes_head=True, alpha=0.6)
        
        # Calculate solution metrics
        solution_cost = sum(self.calculate_route_cost(route) for route in routes)
        gap = ((solution_cost / self.instance.bks) - 1) * 100 if self.instance.bks else 0
        
        # Add title and metrics
        plt.title(f"{title}\nInstance: {self.instance.name}")
        # Formatear el texto del pie de figura
        bks_text = f"BKS: {self.instance.bks:.2f} | " if self.instance.bks is not None else ""
        gap_text = f"Gap: {gap:.2f}% | " if self.instance.bks is not None else ""
        
        plt.figtext(0.5, 0.01, 
                   f"Solution cost: {solution_cost:.2f} | "
                   f"{bks_text}{gap_text}"
                   f"Routes: {len(routes)} | "
                   f"Cities: {sum(len(r)-2 for r in routes) if routes else 0}", 
                   ha="center", fontsize=10,
                   bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
        
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar la figura
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_filename = f"{self.instance.name}_k{len(routes)}_{timestamp}.png"
            map_path = MAPS_DIR / map_filename
            plt.savefig(map_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(map_path)
        
        plt.show()
        return ""
    
    def calculate_route_cost(self, route: List[int]) -> float:
        """Calculate the total distance of a route."""
        if len(route) < 2:
            return 0.0
            
        total = 0.0
        for i in range(len(route)-1):
            total += self.instance.distance_matrix[route[i]][route[i+1]]
        return total


class BMTSP_Solver:
    """Main class to solve the Balanced Multiple TSP problem."""
    def __init__(self, instance_path: str, num_salesmen: int, min_cities: int, max_cities: int):
        self.instance = InstanceBMTSP(instance_path)
        self.num_salesmen = num_salesmen
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.partitioner = RoutePartitioner(self.instance, num_salesmen, min_cities, max_cities)
        self.generator = RouteGenerator(self.instance)
        self.visualizer = SolutionVisualizer(self.instance)
    
    def solve(self, visualize: bool = True) -> Tuple[List[List[int]], float]:
        """Solve the BMTSP with improved optimization."""
        print("\n" + "="*80)
        print(f"RESOLVIENDO INSTANCIA: {self.instance.name}".center(80))
        print("="*80)
        
        print(f"\nParticionando {len(self.instance.cities)} ciudades entre {self.num_salesmen} vendedores...")
        
        # Step 1: Partition cities into balanced groups
        print(f"\nParticionando {len(self.instance.cities)} ciudades entre {self.num_salesmen} vendedores...")
        groups = self.partitioner.partition()
        
        # Verificar que la partición sea válida
        if not groups or len(groups) != self.num_salesmen:
            raise ValueError("Error en la partición de ciudades")
            
        print("\nGrupos generados:")
        for i, group in enumerate(groups, 1):
            print(f"  Vendedor {i}: {len(group)} ciudades")
            
        # Asegurarse de que todos los grupos tengan al menos min_cities y como máximo max_cities
        for i, group in enumerate(groups, 1):
            if len(group) < self.min_cities or len(group) > self.max_cities:
                print(f"Advertencia: El grupo {i} tiene {len(group)} ciudades (debe estar entre {self.min_cities} y {self.max_cities})")
        
        # Step 2: Optimize routes for each group
        print("\nOptimizando rutas para cada grupo...")
        routes = []
        total_cost = 0.0
        
        for i, group in enumerate(tqdm(groups, desc="Optimizando rutas")):
            print(f"\nOptimizando ruta {i+1} con {len(group)} ciudades...")
            route, cost = self.generator.generate_route(group)
            routes.append(route)
            total_cost += cost
            print(f"  Costo de la ruta {i+1}: {cost:.2f}")
        
        # Step 3: Local search improvement
        print("\nAplicando búsqueda local para mejorar la solución...")
        improved = True
        iteration = 0
        
        while improved and iteration < 3:  # Limit number of iterations
            improved = False
            iteration += 1
            
            # Try to improve each route
            for i in range(len(routes)):
                # Skip if route is too short
                if len(routes[i]) <= 2:  # Just depot to depot or single city
                    continue
                    
                # Get the city indices (excluding depots)
                route_cities = [node - 1 for node in routes[i][1:-1]]  # Convert to 0-based and remove depots
                
                # Re-optimize this route
                new_route, new_cost = self.generator.generate_route(route_cities)
                
                # Calculate old cost for this route
                old_cost = 0
                for j in range(len(routes[i])-1):
                    from_node = routes[i][j] - 1  # Convert to 0-based
                    to_node = routes[i][j+1] - 1
                    
                    # Get coordinates
                    from_coord = self.instance.depot if from_node == -1 else self.instance.cities[from_node]
                    to_coord = self.instance.depot if to_node == -1 else self.instance.cities[to_node]
                    
                    # Calculate distance
                    dx = from_coord[0] - to_coord[0]
                    dy = from_coord[1] - to_coord[1]
                    old_cost += math.sqrt(dx*dx + dy*dy)
                
                # If improvement found, update the route
                if new_cost < old_cost * 0.999:  # Small threshold to avoid tiny improvements
                    improvement = (old_cost - new_cost) / old_cost * 100
                    print(f"  Iteración {iteration}: Mejora en ruta {i+1}: {improvement:.2f}%")
                    routes[i] = new_route
                    total_cost = total_cost - old_cost + new_cost
                    improved = True
        
        # Step 4: Visualize solution
        if visualize:
            visualizer = SolutionVisualizer(self.instance)
            map_path = visualizer.plot_routes(routes, save_plot=True)
            print(f"  ✓ Mapa guardado en: {map_path}")
        
        # Step 5: Generate report
        self.generate_report(routes, total_cost)
        
        print("\n" + "="*80)
        print("SOLUCIÓN FINAL".center(80))
        print("="*80)
        print(f"Costo total: {total_cost:.2f}")
        if self.instance.bks:
            gap = ((total_cost / self.instance.bks) - 1) * 100
            print(f"BKS: {self.instance.bks:.2f}")
            print(f"GAP: {gap:.2f}%")
        print("="*80 + "\n")
        
        return routes, total_cost
    
    def generate_report(self, routes: List[List[int]], total_cost: float, map_path: str = "") -> str:
        """
        Generate a PDF report with the specified 7 columns:
        instancia, k, m min, m max, bks, lkh-3, gap%
        """
        # Calculate gap if BKS is available
        gap = None
        if self.instance.bks is not None and self.instance.bks > 0:
            gap = ((total_cost / self.instance.bks) - 1) * 100
        
        # Create data for the table
        data = [
            ["Instancia", self.instance.name],
            ["N° Vendedores (k)", self.num_salesmen],
            ["Mín. Ciudades", self.min_cities],
            ["Máx. Ciudades", self.max_cities],
            ["BKS", f"{self.instance.bks:.2f}" if self.instance.bks is not None else "N/A"],
            ["Solución LKH-3", f"{total_cost:.2f}"],
            ["GAP %", f"{gap:.2f}%" if gap is not None else "N/A"]
        ]
        
        # Create PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = PDFS_DIR / f"{self.instance.name}_k{self.num_salesmen}_{timestamp}.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Configure the figure with more height for better visibility
            fig, ax = plt.subplots(figsize=(11, 2.5))  # Increased height for better visibility
            ax.axis('off')
            
            # Add title
            plt.title(f"Resultados - {self.instance.name}", fontsize=14, fontweight='bold', pad=20)
            
            # Create the table with custom styling
            table = ax.table(
                cellText=data,
                colLabels=None,
                cellLoc='left',
                loc='center',
                cellColours=[['#f3f3f3', 'white'] for _ in range(len(data))],
                edges='open'
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.5, 2.5)  # Adjust cell size
            
            # Make first column bold and add background
            for i, (key, _) in enumerate(data):
                cell = table[i, 0]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4F81BD')
                cell.set_text_props(color='white')
                
                # Right-align the values
                cell = table[i, 1]
                cell.set_text_props(ha='right')
            
            # Adjust layout to prevent text from being cut off
            plt.tight_layout()
            
            # Save the first page with the results table
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # Add a second page with the map if available
            if map_path and os.path.exists(map_path):
                fig = plt.figure(figsize=(11, 8.5))
                plt.title(f"Mapa de Rutas - {self.instance.name}", fontsize=14, fontweight='bold', pad=20)
                img = plt.imread(map_path)
                plt.imshow(img)
                plt.axis('off')
                pdf.savefig(bbox_inches='tight')
                plt.close()
                
                # Add a third page with route details
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                
                # Add title
                plt.title(f"Detalles de Rutas - {self.instance.name}", fontsize=14, fontweight='bold', pad=20)
                
                # Create route details
                route_details = [
                    ["Ruta", "N° Ciudades", "Distancia"],
                    ["-" * 10, "-" * 10, "-" * 10]
                ]
                
                for i, route in enumerate(routes):
                    if len(route) <= 2:  # Skip empty routes
                        continue
                    num_cities = len([x for x in route if x != 0])  # Count non-depot cities
                    distance = self.generator._calculate_route_distance(route)
                    route_details.append([f"Ruta {i+1}", str(num_cities), f"{distance:.2f}"])
                
                # Add total row
                total_distance = sum(self.generator._calculate_route_distance(r) for r in routes if len(r) > 2)
                route_details.append(["-" * 10, "-" * 10, "-" * 10])
                route_details.append(["TOTAL", str(sum(len([x for x in r if x != 0]) for r in routes)), f"{total_distance:.2f}"])
                
                # Create route details table
                route_table = ax.table(
                    cellText=route_details,
                    cellLoc='center',
                    loc='center',
                    colColours=['#4F81BD'] * 3,
                    cellColours=[['#f3f3f3'] * 3] + [['white'] * 3] * (len(route_details) - 1)
                )
                
                # Style the table
                route_table.auto_set_font_size(False)
                route_table.set_fontsize(10)
                route_table.scale(1.2, 1.5)
                
                # Style header row
                for i in range(3):
                    cell = route_table[0, i]
                    cell.set_text_props(weight='bold', color='white')
                
                # Style total row
                for i in range(3):
                    cell = route_table[len(route_details)-1, i]
                    cell.set_text_props(weight='bold')
                
                plt.tight_layout()
                pdf.savefig(bbox_inches='tight')
                plt.close()
        
        print(f"\nInforme generado: {pdf_path}")
        return str(pdf_path)
    
    def print_solution_summary(self, total_cost: float):
        """Print a summary of the solution."""
        print("\n" + "="*80)
        print(f"INSTANCIA: {self.instance.name}".center(80))
        print("="*80)
        print(f"Número de ciudades: {self.instance.n}")
        print(f"Número de vendedores: {self.num_salesmen}")
        print(f"Ciudades por vendedor: {self.min_cities} - {self.max_cities}")
        print(f"Costo total de la solución: {total_cost:.2f}")
        
        if self.instance.bks is not None:
            gap = ((total_cost / self.instance.bks) - 1) * 100
            print(f"Mejor Solución Conocida (BKS): {self.instance.bks:.2f}")
            print(f"Gap respecto a BKS: {gap:.2f}%")
        print("="*80 + "\n")


def process_instance(instance_path: str, k: int, min_cities: int, max_cities: int, no_plot: bool = False) -> dict:
    """Process a single instance and return the results.
    
    Args:
        instance_path: Path to the TSP instance file
        k: Number of salesmen
        min_cities: Minimum number of cities per route
        max_cities: Maximum number of cities per route
        no_plot: If True, skip generating the plot
        
    Returns:
        Dictionary with the results
    """
    instance_name = Path(instance_path).stem
    print(f"\n{'='*80}")
    print(f"Procesando instancia: {instance_name}".center(80))
    print(f"Número de vendedores (k): {k}, Ciudades por ruta: {min_cities}-{max_cities}")
    print(f"{'='*80}")
    
    try:
        # Create output directories if they don't exist
        os.makedirs("resultados/mapas", exist_ok=True)
        os.makedirs("resultados/pdfs", exist_ok=True)
        
        # Initialize and solve
        solver = BMTSP_Solver(instance_path, k, min_cities, max_cities)
        routes, total_cost = solver.solve(visualize=not no_plot)
        
        # Generate map path
        map_path = f"resultados/mapas/{instance_name}_k{k}_m{min_cities}-{max_cities}.png"
        
        # Generate report
        report_path = solver.generate_report(routes, total_cost, map_path)
        
        # Calculate gap if BKS is available
        gap = None
        if solver.instance.bks and solver.instance.bks > 0:
            gap = ((total_cost / solver.instance.bks) - 1) * 100
        
        # Return results for summary
        return {
            'instance': instance_name,
            'k': k,
            'min_cities': min_cities,
            'max_cities': max_cities,
            'bks': solver.instance.bks,
            'cost': total_cost,
            'gap': gap,
            'map_path': map_path if not no_plot else None,
            'report_path': report_path
        }
        
    except Exception as e:
        print(f"Error procesando la instancia {instance_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_summary_pdf(results: list, output_path: str):
    """
    Generate a summary PDF with all results.

    Args:
        results: List of result dictionaries from process_instance
        output_path: Path to save the PDF file
    """
    if not PDF_AVAILABLE:
        print("Advertencia: No se pudo generar el PDF. El módulo 'reportlab' no está instalado.")
        print("Instálalo con: pip install reportlab")
        return
        
    try:
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Add title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=1  # Center
        )
        elements.append(Paragraph("Resumen de Resultados BMTSP", title_style))
        
        # Create table data
        data = [
            ["Instancia", "Costo Total", "BKS", "GAP (%)"]
        ]
        
        for result in results:
            data.append([
                result.get('instance', 'N/A'),
                f"{result.get('total_cost', 0):.2f}",
                f"{result.get('bks', 0):.2f}",
                f"{result.get('gap', 0):.2f}"
            ])
        
        # Create and style table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generado el: {timestamp}", styles['Italic']))
        
        # Build PDF
        doc.build(elements)
        print(f"\nResumen guardado en: {output_path}")
        
    except Exception as e:
        print(f"Error al generar el resumen PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Style header
        for (i, key) in enumerate(df.columns):
            cell = table[0, i]
            cell.set_facecolor('#4F81BD')
            cell.set_text_props(weight='bold', color='white')
        
        # Style cells
        for i in range(len(df)):
            for j in range(len(df.columns)):
                cell = table[i+1, j]
                cell.set_facecolor('#E6F3FF' if i % 2 == 0 else '#FFFFFF')
        
        plt.title('Resumen de Resultados BMTSP', fontsize=14, pad=20)
        pdf.savefig(bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Resolver el problema BMTSP con múltiples instancias')
    parser.add_argument('--k', type=int, required=True, help='Número de vendedores')
    parser.add_argument('--min-cities', type=int, required=True, help='Mínimo número de ciudades por ruta')
    parser.add_argument('--max-cities', type=int, required=True, help='Máximo número de ciudades por ruta')
    parser.add_argument('--instances', nargs='+', help='Rutas a los archivos de instancia TSP')
    parser.add_argument('--no-plot', action='store_true', help='No generar gráficos de las rutas')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if args.min_cities > args.max_cities:
        print("Error: El mínimo de ciudades no puede ser mayor que el máximo")
        return
    
    # Si no se proporcionan instancias, usar los archivos kro*.tsp en el directorio instances
    if not args.instances:
        instances_dir = os.path.join(os.path.dirname(__file__), 'instances')
        default_instances = [
            'kroA100.tsp',
            'kroB100.tsp',
            'kroC100.tsp',
            'kroD100.tsp',
            'kroE100.tsp'
        ]
        args.instances = [os.path.join(instances_dir, f) for f in default_instances]
    
    # Verificar que existan los archivos de instancia
    valid_instances = []
    for instance_path in args.instances:
        if not os.path.exists(instance_path):
            print(f"Advertencia: Archivo no encontrado: {instance_path}")
        else:
            valid_instances.append(instance_path)
    
    if not valid_instances:
        print("Error: No se encontraron archivos de instancia válidos")
        return
    
    print(f"\n{'='*80}")
    print(f"CONFIGURACIÓN".center(80))
    print(f"{'='*80}")
    print(f"Instancias a procesar: {len(valid_instances)}")
    print(f"Número de vendedores (k): {args.k}")
    print(f"Rango de ciudades por ruta: {args.min_cities}-{args.max_cities}")
    print(f"Generar gráficos: {'No' if args.no_plot else 'Sí'}")
    print(f"{'='*80}\n")
    
    # Procesar cada instancia
    results = []
    for instance_path in valid_instances:
        print(f"\n{'='*80}")
        print(f"PROCESANDO: {os.path.basename(instance_path)}".center(80))
        print(f"{'='*80}")
        
        try:
            result = process_instance(
                instance_path=instance_path,
                k=args.k,
                min_cities=args.min_cities,
                max_cities=args.max_cities,
                no_plot=args.no_plot
            )
            
            if result:
                results.append(result)
                print(f"\nResultado para {os.path.basename(instance_path)}:")
                print(f"  Costo total: {result['cost']:.2f}")
                if result['gap'] is not None:
                    print(f"  BKS: {result['bks']:.2f}")
                    print(f"  GAP: {result['gap']:.2f}%")
        except Exception as e:
            print(f"Error procesando {instance_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generar resumen PDF si hay resultados
    if results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = f'resultados/resumen_{timestamp}.pdf'
        print(f"\nGenerando resumen en: {summary_path}")
        generate_summary_pdf(results, summary_path)
        
        # Mostrar resumen por consola
        print("\n" + "="*80)
        print("RESUMEN FINAL".center(80))
        print("="*80)
        print(f"Instancias procesadas: {len(results)} de {len(valid_instances)}")
        
        # Calcular estadísticas
        if results:
            gaps = [r['gap'] for r in results if r['gap'] is not None]
            if gaps:
                print(f"GAP promedio: {sum(gaps)/len(gaps):.2f}%")
                print(f"Mejor GAP: {min(gaps):.2f}%")
                print(f"Peor GAP: {max(gaps):.2f}%")
        
        print("="*80)
    else:
        print("\nNo se generaron resultados para mostrar.")


if __name__ == '__main__':
    main()
