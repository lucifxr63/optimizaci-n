import math
import random
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class City:
    idx: int
    x: float
    y: float

    def distance_to(self, other: "City") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

def get_tsp_files():
    """Get the list of Kro*100.tsp files from LKH3.0/instances/"""
    instances_dir = os.path.join('..', 'LKH3.0', 'instances')
    tsp_files = [
        'kroA100.tsp',
        'kroB100.tsp',
        'kroC100.tsp',
        'kroD100.tsp',
        'kroE100.tsp'
    ]
    
    # Check which files exist
    existing_files = []
    for file in tsp_files:
        file_path = os.path.join(instances_dir, file)
        if os.path.exists(file_path):
            existing_files.append((file, file_path))
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not existing_files:
        print("No TSP files found. Please check the directory:")
        print(os.path.abspath(instances_dir))
    
    return existing_files

def load_tsp_file(file_path: str) -> List[City]:
    """Load cities from a TSPLIB format file."""
    cities = []
    with open(file_path, 'r') as f:
        # Skip header lines until NODE_COORD_SECTION
        line = f.readline().strip()
        while line and 'NODE_COORD_SECTION' not in line:
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
                    cities.append(City(idx, x, y))
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not parse line: {line} - {e}")
                    continue
    return cities

def nearest_neighbor_route(cities: List[City], start: City) -> List[City]:
    """Create a route using the nearest neighbor heuristic."""
    if not cities:
        return []
        
    route = [start]
    remaining = [c for c in cities if c != start]
    current = start
    
    while remaining:
        next_city = min(remaining, key=lambda c: current.distance_to(c))
        route.append(next_city)
        remaining.remove(next_city)
        current = next_city
    
    route.append(start)  # Return to start
    return route

def route_length(route: List[City]) -> float:
    """Calculate the total length of a route."""
    if len(route) < 2:
        return 0.0
    return sum(route[i].distance_to(route[i+1]) for i in range(len(route)-1))

def two_opt(route: List[City]) -> List[City]:
    """Improve the route using the 2-opt algorithm."""
    if len(route) <= 2:
        return route
        
    best = route
    improved = True
    
    while improved:
        improved = False
        best_distance = route_length(best)
        
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue
                # Create new route with the swap
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_distance = route_length(new_route)
                if new_distance < best_distance:
                    best = new_route
                    best_distance = new_distance
                    improved = True
        
        if improved:
            route = best
    
    return best

def assign_cities(cities: List[City], k: int, max_cities: int) -> List[List[City]]:
    """Assign cities to k salesmen with a maximum number of cities each."""
    if not cities:
        return [[] for _ in range(k)]
        
    depot = cities[0]  # First city is the depot
    other_cities = [c for c in cities if c != depot]
    random.shuffle(other_cities)
    
    # Distribute cities among salesmen
    assignments = [[] for _ in range(k)]
    for i, city in enumerate(other_cities):
        assignments[i % k].append(city)
    
    # Limit the number of cities per salesman
    for i in range(k):
        assignments[i] = assignments[i][:max_cities]
    
    return assignments

def heuristic_bmtsp(cities: List[City], k: int, max_cities: int) -> Tuple[List[List[City]], float]:
    """Solve the BMTSP using a heuristic approach."""
    if not cities:
        return [], 0.0
        
    depot = cities[0]
    assignments = assign_cities(cities, k, max_cities)
    routes = []
    total_cost = 0.0
    
    for assigned in assignments:
        if not assigned:
            routes.append([depot, depot])
            continue
            
        # Add depot at start and end of the route
        route = [depot] + assigned + [depot]
        # Improve the route
        route = two_opt(route)
        routes.append(route)
        total_cost += route_length(route)
    
    return routes, total_cost

import time
import statistics

def calculate_route_metrics(routes: List[List[City]]) -> dict:
    """Calculate various metrics for the routes."""
    route_lengths = [route_length(route) for route in routes]
    num_cities = [len(route) - 2 for route in routes]  # -2 for depot at start/end
    
    return {
        'total_cost': sum(route_lengths),
        'num_routes': len(routes),
        'avg_route_length': statistics.mean(route_lengths) if route_lengths else 0,
        'min_route_length': min(route_lengths) if route_lengths else 0,
        'max_route_length': max(route_lengths) if route_lengths else 0,
        'std_dev_length': statistics.stdev(route_lengths) if len(route_lengths) > 1 else 0,
        'avg_cities_per_route': statistics.mean(num_cities) if num_cities else 0,
        'min_cities': min(num_cities) if num_cities else 0,
        'max_cities': max(num_cities) if num_cities else 0,
        'execution_time': 0  # Will be set by the main function
    }

def save_routes_to_file(routes: List[List[City]], total_cost: float, filename: str, metrics: dict = None, problem_name: str = ""):
    """Save the routes to a file with LKH-compatible format and detailed metrics."""
    if metrics is None:
        metrics = calculate_route_metrics(routes)
    
    with open(filename, 'w') as f:
        # LKH-compatible header
        f.write(f"NAME: {problem_name or 'BMTSP_SOLUTION'}\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {sum(len(r)-1 for r in routes) + 1}\n")  # -1 for repeated depot
        f.write(f"NUMBER_OF_VEHICLES: {len(routes)}\n")
        f.write(f"TOTAL_COST: {metrics['total_cost']:.2f}\n")
        f.write(f"DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("TOUR_SECTION\n")
        
        # Write routes
        for i, route in enumerate(routes, 1):
            route_str = ' '.join(str(c.idx) for c in route[:-1])  # Exclude last depot
            f.write(f"#{i} {route_str}\n")
        
        f.write("-1\n")
        f.write("EOF\n\n")
        
        # Detailed statistics
        f.write("DETAILED_STATISTICS\n")
        f.write("="*60 + "\n")
        f.write(f"{'METRIC':<30} | {'VALUE':>25}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Total Cost':<30} | {metrics['total_cost']:>25.2f}\n")
        f.write(f"{'Number of Routes':<30} | {metrics['num_routes']:>25}\n")
        f.write(f"{'Average Route Length':<30} | {metrics['avg_route_length']:>25.2f}\n")
        f.write(f"{'Shortest Route':<30} | {metrics['min_route_length']:>25.2f}\n")
        f.write(f"{'Longest Route':<30} | {metrics['max_route_length']:>25.2f}\n")
        f.write(f"{'Std Dev of Route Lengths':<30} | {metrics['std_dev_length']:>25.2f}\n")
        f.write(f"{'Avg Cities per Route':<30} | {metrics['avg_cities_per_route']:>25.2f}\n")
        f.write(f"{'Min Cities in Route':<30} | {metrics['min_cities']:>25}\n")
        f.write(f"{'Max Cities in Route':<30} | {metrics['max_cities']:>25}\n")
        f.write(f"{'Execution Time (s)':<30} | {metrics['execution_time']:>25.2f}\n")
        
        # Individual route details
        f.write("\n" + "INDIVIDUAL ROUTE DETAILS".center(60, '=') + "\n\n")
        f.write(f"{'VEHICLE':<8} | {'CITIES':<8} | {'DISTANCE':>12} | {'ROUTE'}\n")
        f.write("-"*80 + "\n")
        
        for i, route in enumerate(routes, 1):
            num_cities = len(route) - 2  # Exclude depot at start/end
            distance = route_length(route)
            route_str = ' -> '.join(str(c.idx) for c in route)
            f.write(f"{i:<8} | {num_cities:<8} | {distance:>12.2f} | {route_str}\n")

def main():
    # Get list of TSP files to process
    tsp_files = get_tsp_files()
    if not tsp_files:
        print("No TSP files found. Exiting...")
        return
    
    k = 5  # Number of salesmen
    max_cities = 20  # Maximum cities per salesman
    
    # Create results directory if it doesn't exist
    os.makedirs('resultados', exist_ok=True)
    
    # Create a timestamp for the summary file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join('resultados', f'summary_{timestamp}.txt')
    
    print(f"Starting BMTSP solver with {k} salesmen and max {max_cities} cities per salesman\n")
    
    # Initialize summary data
    all_metrics = []
    
    # Write summary header
    with open(summary_file, 'w') as sf:
        sf.write("BMTSP HEURISTIC - COMPARATIVE RESULTS\n")
        sf.write("="*60 + "\n")
        sf.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        sf.write(f"Number of salesmen: {k}\n")
        sf.write(f"Max cities per salesman: {max_cities}\n\n")
        sf.write(f"{'PROBLEM':<15} | {'TOTAL COST':>12} | {'BEST ROUTE':>12} | "
                f"{'WORST ROUTE':>12} | {'STD DEV':>8} | {'TIME (s)':>8}\n")
        sf.write("-"*85 + "\n")
    
    for tsp_file, file_path in tsp_files:
        if not os.path.exists(file_path):
            print(f"File {tsp_file} not found at {file_path}, skipping...")
            continue
            
        problem_name = os.path.splitext(tsp_file)[0]
        print(f"\n{'='*60}")
        print(f"PROCESSING: {tsp_file}")
        print(f"{'='*60}")
        
        try:
            # Load cities from TSP file
            print("  Loading cities...")
            start_time = time.time()
            cities = load_tsp_file(file_path)
            if not cities:
                print(f"  Error: No cities loaded from {tsp_file}")
                continue
                
            print(f"  Loaded {len(cities)} cities")
            
            # Solve BMTSP
            print(f"  Solving BMTSP with k={k}, max_cities={max_cities}...")
            routes, total_cost = heuristic_bmtsp(cities, k, max_cities)
            exec_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_route_metrics(routes)
            metrics['execution_time'] = exec_time
            all_metrics.append((problem_name, metrics))
            
            # Print summary
            print("\n  === SOLUTION SUMMARY ===")
            print(f"  Total cost: {metrics['total_cost']:.2f}")
            print(f"  Best route: {metrics['min_route_length']:.2f}")
            print(f"  Worst route: {metrics['max_route_length']:.2f}")
            print(f"  StdDev: {metrics['std_dev_length']:.2f}")
            print(f"  Execution time: {exec_time:.2f} seconds")
            
            # Save detailed results
            output_file = os.path.join('resultados', f"{problem_name}_result.txt")
            save_routes_to_file(routes, total_cost, output_file, metrics, problem_name)
            print(f"\n  Detailed routes saved to: {output_file}")
            
            # Update summary file
            with open(summary_file, 'a') as sf:
                sf.write(
                    f"{problem_name:<15} | {metrics['total_cost']:>12.2f} | "
                    f"{metrics['min_route_length']:>12.2f} | "
                    f"{metrics['max_route_length']:>12.2f} | "
                    f"{metrics['std_dev_length']:>8.2f} | "
                    f"{exec_time:>8.2f}\n"
                )
            
        except Exception as e:
            print(f"  Error processing {tsp_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if all_metrics:
        print("\n" + "="*60)
        print("EXECUTION COMPLETED")
        print("="*60)
        print(f"Processed {len(all_metrics)} problems")
        print(f"Summary file: {os.path.abspath(summary_file)}")
    else:
        print("\nNo problems were processed successfully.")

if __name__ == "__main__":
    main()
