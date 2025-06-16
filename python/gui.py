import os
import re
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

from heuristica_bmtsp import load_cities, heuristic_bmtsp, plot_routes


def append_output(widget, text):
    widget.insert(tk.END, text)
    widget.see(tk.END)


def parse_ampl_routes(ampl_text, cities):
    """Parse routes from AMPL output displaying x[s,i,j] variables."""
    pattern = re.compile(r"x\[(\d+),(\d+),(\d+)\]\s*=\s*1")
    edges = pattern.findall(ampl_text)
    if not edges:
        return []
    edges = [(int(s), int(i), int(j)) for s, i, j in edges]
    salesmen = sorted({s for s, _, _ in edges})
    by_s = {s: {} for s in salesmen}
    for s, i, j in edges:
        by_s[s][i] = j
    city_map = {c.idx: c for c in cities}
    routes = []
    for s in salesmen:
        route = [city_map[0]]
        current = 0
        while True:
            nxt = by_s[s].get(current)
            if nxt is None:
                break
            route.append(city_map[nxt])
            if nxt == 0:
                break
            current = nxt
        routes.append(route)
    return routes


def parse_lkh_tour(tour_file, cities):
    """Parse LKH tour file into routes."""
    if not os.path.exists(tour_file):
        return []
    numbers = []
    with open(tour_file) as f:
        for tok in f.read().split():
            try:
                numbers.append(int(tok))
            except ValueError:
                pass
    if not numbers:
        return []
    seq = numbers[1:]
    city_map = {c.idx: c for c in cities}
    routes = []
    route = [city_map[0]]
    for n in seq:
        if n == -1:
            route.append(city_map[0])
            routes.append(route)
            route = [city_map[0]]
        elif n == 0:
            break
        else:
            route.append(city_map[n - 1])
    if len(route) > 1:
        route.append(city_map[0])
        routes.append(route)
    return routes


def run_ampl(output, csv_path):
    if not shutil.which("ampl"):
        append_output(output, "AMPL executable not found in PATH\n")
        return
    cmd = (
        "model ampl/modelo_bmtsp.mod; data ampl/datos_bmtsp.dat; solve;"
        " display _solve_message;"
        " display {s in S, i in V, j in V: x[s,i,j] > 0.5} x[s,i,j];"
    )
    try:
        result = subprocess.run(
            ["ampl"], input=cmd, text=True, capture_output=True, check=True
        )
        append_output(output, result.stdout + "\n")
        cities = load_cities(csv_path.get())
        routes = parse_ampl_routes(result.stdout, cities)
        if routes:
            plot_routes(routes)
    except subprocess.CalledProcessError as e:
        append_output(output, e.stderr + "\n")


def run_lkh(output, csv_path):
    if not shutil.which("LKH"):
        append_output(output, "LKH executable not found in PATH\n")
        return
    try:
        result = subprocess.run(
            ["LKH", "lkh/example.par"], text=True, capture_output=True, check=True
        )
        append_output(output, result.stdout + "\n")
        tour_file = os.path.join("lkh", "example.tour")
        cities = load_cities(csv_path.get())
        routes = parse_lkh_tour(tour_file, cities)
        if routes:
            plot_routes(routes)
    except subprocess.CalledProcessError as e:
        append_output(output, e.stderr + "\n")


def run_heuristic(csv_path, k_entry, max_entry, output):
    try:
        k = int(k_entry.get())
        max_c = int(max_entry.get())
    except ValueError:
        append_output(output, "Invalid numeric parameters\n")
        return
    if not os.path.exists(csv_path.get()):
        append_output(output, "CSV file not found\n")
        return
    cities = load_cities(csv_path.get())
    routes, cost = heuristic_bmtsp(cities, k, max_c)
    append_output(output, f"Total cost: {cost:.2f}\n")
    for i, route in enumerate(routes, 1):
        path = " -> ".join(str(c.idx) for c in route)
        append_output(output, f"Salesman {i}: {path}\n")
    append_output(output, "\n")
    plot_routes(routes)


def main():
    root = tk.Tk()
    root.title("BMTSP Interface")

    frm = ttk.Frame(root, padding=10)
    frm.grid()

    ttk.Label(frm, text="CSV file:").grid(column=0, row=0, sticky=tk.W)
    csv_var = tk.StringVar(value="python/ciudades.csv")
    csv_entry = ttk.Entry(frm, width=30, textvariable=csv_var)
    csv_entry.grid(column=1, row=0, columnspan=2, sticky=tk.W)

    ttk.Label(frm, text="Salesmen (k):").grid(column=0, row=1, sticky=tk.W)
    k_entry = ttk.Entry(frm, width=5)
    k_entry.insert(0, "2")
    k_entry.grid(column=1, row=1, sticky=tk.W)

    ttk.Label(frm, text="Max cities:").grid(column=0, row=2, sticky=tk.W)
    max_entry = ttk.Entry(frm, width=5)
    max_entry.insert(0, "5")
    max_entry.grid(column=1, row=2, sticky=tk.W)

    output = scrolledtext.ScrolledText(frm, width=60, height=20)
    output.grid(column=0, row=4, columnspan=3, pady=10)

    ttk.Button(
        frm,
        text="Run Heuristic",
        command=lambda: run_heuristic(csv_var, k_entry, max_entry, output),
    ).grid(column=0, row=3, sticky=tk.W)

    ttk.Button(
        frm, text="Run AMPL", command=lambda: run_ampl(output, csv_var)
    ).grid(column=1, row=3, sticky=tk.W)

    ttk.Button(
        frm, text="Run LKH", command=lambda: run_lkh(output, csv_var)
    ).grid(column=2, row=3, sticky=tk.W)

    root.mainloop()


if __name__ == "__main__":
    main()
