"""Heuristic solution for the Bounded Multiple Traveling Salesman Problem.

Given a file ``ciudades.csv`` with city coordinates, this script assigns
cities to ``k`` salesmen randomly (respecting ``max_ciudades``) and
computes a route for each salesman using nearest neighbor plus 2-opt
local optimization.
"""

import csv
import math
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is optional
    plt = None

@dataclass
class City:
    idx: int
    x: float
    y: float

    def distance_to(self, other: "City") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def load_cities(path: str) -> List[City]:
    """Load cities from CSV file."""
    cities = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append(City(int(row["id"]), float(row["x"]), float(row["y"])))
    # Sort by id to ensure depot is first
    cities.sort(key=lambda c: c.idx)
    return cities


def nearest_neighbor_route(cities: List[City], start: City) -> List[City]:
    """Return a route starting from ``start`` visiting all ``cities`` using
    the nearest neighbor heuristic."""
    route = [start]
    remaining = cities[:]
    current = start
    while remaining:
        next_city = min(remaining, key=lambda c: current.distance_to(c))
        route.append(next_city)
        remaining.remove(next_city)
        current = next_city
    route.append(start)  # return to depot
    return route


def two_opt(route: List[City]) -> List[City]:
    """Simple 2-opt improvement for a single route."""
    improved = True
    best = route
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if route_length(new_route) < route_length(best):
                    best = new_route
                    improved = True
        route = best
    return best


def route_length(route: List[City]) -> float:
    return sum(route[i].distance_to(route[i + 1]) for i in range(len(route) - 1))


def assign_cities_randomly(cities: List[City], k: int, max_ciudades: int) -> List[List[City]]:
    """Assign cities (except depot) randomly to k salesmen respecting capacity."""
    depot = cities[0]
    others = cities[1:]
    random.shuffle(others)
    assignments = [[] for _ in range(k)]
    idx = 0
    for city in others:
        assignments[idx % k].append(city)
        idx += 1
    # Enforce max_ciudades by truncation
    for i in range(k):
        assignments[i] = assignments[i][:max_ciudades]
    return assignments


def heuristic_bmtsp(cities: List[City], k: int, max_ciudades: int) -> Tuple[List[List[City]], float]:
    depot = cities[0]
    assignments = assign_cities_randomly(cities, k, max_ciudades)
    routes = []
    total_cost = 0.0
    for assigned in assignments:
        if not assigned:
            routes.append([depot, depot])
            continue
        route = nearest_neighbor_route(assigned, depot)
        route = two_opt(route)
        routes.append(route)
        total_cost += route_length(route)
    return routes, total_cost


def plot_routes(routes: List[List[City]]):
    if plt is None:
        print("matplotlib not installed; skipping plot")
        return
    colors = [f"C{i}" for i in range(len(routes))]
    for color, route in zip(colors, routes):
        xs = [c.x for c in route]
        ys = [c.y for c in route]
        plt.plot(xs, ys, marker="o", color=color)
        plt.annotate(str(route[0].idx), (route[0].x, route[0].y))
    plt.title("Heuristic BMTSP Routes")
    plt.show()


def main(args):
    if len(args) < 3:
        print("Usage: python heuristica_bmtsp.py ciudades.csv k max_ciudades")
        return
    csv_path = args[0]
    k = int(args[1])
    max_ciudades = int(args[2])
    cities = load_cities(csv_path)
    routes, cost = heuristic_bmtsp(cities, k, max_ciudades)
    print(f"Costo total: {cost:.2f}")
    for i, route in enumerate(routes, 1):
        path = " -> ".join(str(c.idx) for c in route)
        print(f"Vendedor {i}: {path}")
    plot_routes(routes)


if __name__ == "__main__":
    main(sys.argv[1:])
