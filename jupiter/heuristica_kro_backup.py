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

def save_routes_to_file(routes: List[List[City]], total_cost: float, filename: str):
    """Save the routes to a file."""
    with open(filename, 'w') as f:
        f.write(f"Total cost: {total_cost:.2f}\n\n")
        for i, route in enumerate(routes, 1):
            f.write(f"Salesman {i} route (length: {route_length(route):.2f}):\n")
            f.write(" -> ".join(str(c.idx) for c in route) + "\n\n")

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
    
    print(f"Starting BMTSP solver with {k} salesmen and max {max_cities} cities per salesman\n")
    
    for tsp_file, file_path in tsp_files:
        if not os.path.exists(file_path):
            print(f"File {tsp_file} not found at {file_path}, skipping...")
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing {tsp_file}...")
        print(f"{'='*50}")
        
        try:
            # Load cities from TSP file
            print("  Loading cities...")
            cities = load_tsp_file(file_path)
            if not cities:
                print(f"  Error: No cities loaded from {tsp_file}")
                continue
                
            print(f"  Loaded {len(cities)} cities")
            
            # Solve BMTSP
            print(f"  Solving BMTSP with k={k}, max_cities={max_cities}...")
            routes, total_cost = heuristic_bmtsp(cities, k, max_cities)
            
            # Print results
            print(f"\n  Results for {tsp_file}:")
            print(f"  {'-'*40}")
            print(f"  Total cost: {total_cost:.2f}")
            print(f"  Number of routes: {len(routes)}")
            
            # Print route summaries
            for i, route in enumerate(routes, 1):
                print(f"  - Salesman {i}: {len(route)-2} cities, "
                      f"route length: {route_length(route):.2f}")
            
            # Save detailed routes to file
            output_file = os.path.join('resultados', f"{os.path.splitext(tsp_file)[0]}_result.txt")
            save_routes_to_file(routes, total_cost, output_file)
            print(f"\n  Detailed routes saved to: {output_file}")
            
        except Exception as e:
            print(f"  Error processing {tsp_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
