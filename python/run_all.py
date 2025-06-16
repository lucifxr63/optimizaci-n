"""Runner script to execute heuristic and optionally LKH for BMTSP."""

import os
import subprocess
import shutil
import sys
from pathlib import Path

from heuristica_bmtsp import load_cities, heuristic_bmtsp, plot_routes


def run_heuristic(csv_path: str, k: int, max_ciudades: int):
    cities = load_cities(csv_path)
    routes, cost = heuristic_bmtsp(cities, k, max_ciudades)
    print("\n== Heuristic Solution ==")
    print(f"Total cost: {cost:.2f}")
    for i, route in enumerate(routes, 1):
        path = " -> ".join(str(c.idx) for c in route)
        print(f"Salesman {i}: {path}")
    plot_routes(routes)


def run_lkh(tsp_file: str, par_file: str):
    print("\n== LKH Solution ==")
    if not shutil.which("LKH" ):
        print("LKH executable not found in PATH")
        return
    subprocess.run(["LKH", par_file], check=True)
    tour_file = Path(par_file).with_suffix(".tour")
    if tour_file.exists():
        with open(tour_file) as f:
            print(f.read())
    else:
        print("Tour file not found")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_all.py ciudades.csv k max_ciudades [--lkh tsp par]")
        sys.exit(1)

    csv_path = sys.argv[1]
    k = int(sys.argv[2])
    max_ciudades = int(sys.argv[3])

    run_heuristic(csv_path, k, max_ciudades)

    if "--lkh" in sys.argv:
        idx = sys.argv.index("--lkh")
        try:
            tsp_file = sys.argv[idx + 1]
            par_file = sys.argv[idx + 2]
        except IndexError:
            print("Specify tsp and par file after --lkh")
            sys.exit(1)
        run_lkh(tsp_file, par_file)
