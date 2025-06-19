import os
import subprocess
import math
import csv
import time
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg for non-interactive plotting
import matplotlib.pyplot as plt
from fpdf import FPDF
import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path

class Ciudad:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = float(x)
        self.y = float(y)

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

def load_cities(tsp_path):
    cities = []
    with open(tsp_path, 'r') as f:
        lines = f.readlines()
        start = lines.index("NODE_COORD_SECTION\n") + 1
        for line in lines[start:]:
            if line.strip() == "EOF":
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                cities.append(Ciudad(int(parts[0]) - 1, parts[1], parts[2]))
    if not cities:
        raise ValueError(f"No se cargaron ciudades desde '{tsp_path}'")
    return cities

def parse_lkh_tour(tour_file):
    with open(tour_file, 'r') as f:
        lines = f.readlines()

    tours = []
    cost = 0.0

    for line in lines:
        line = line.strip()

        if "Cost:" in line:
            try:
                value = line.split("Cost:")[1].strip().replace("_", "").replace(",", "")
                cost = float(value)
            except (IndexError, ValueError):
                cost = 0.0

        if line.startswith("1") and "(#" in line:
            try:
                route_str = line.split("(#")[0].strip()
                nodes = [int(x) - 1 for x in route_str.split() if x.isdigit()]
                if nodes and nodes[0] == 0:
                    nodes = nodes[1:]
                if nodes and nodes[-1] == 0:
                    nodes = nodes[:-1]
                tours.append(nodes)
            except Exception:
                pass

    return cost, tours

def export_routes_to_csv(rutas, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Vendedor", "Ruta"])
        for i, ruta in enumerate(rutas, 1):
            writer.writerow([f"Vendedor {i}", "0 -> " + " -> ".join(str(n) for n in ruta) + " -> 0"])

def plot_route_map(cities, routes, output_path):
    """
    Plot and save a map of the routes.
    
    Args:
        cities: List of City objects
        routes: List of routes, where each route is a list of city indices
        output_path: Path to save the output image
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    plt.figure(figsize=(12, 10))
    
    # Plot all cities
    x_coords = [city.x for city in cities[1:]]  # Skip depot for individual points
    y_coords = [city.y for city in cities[1:]]
    plt.scatter(x_coords, y_coords, color='#1f77b4', s=80, alpha=0.7, 
               edgecolors='white', linewidth=1, label='Ciudades')
    
    # Plot depot
    plt.scatter([cities[0].x], [cities[0].y], color='black', marker='s', 
               s=150, label='Depósito', zorder=10, edgecolor='white', linewidth=1.5)
    
    # Plot each route with a different color
    for i, route in enumerate(routes):
        if not route:
            continue
            
        color = colors[i % len(colors)]
        # Include depot at start and end of route
        route_coords = [(cities[0].x, cities[0].y)]
        for node_idx in route:
            route_coords.append((cities[node_idx].x, cities[node_idx].y))
        route_coords.append((cities[0].x, cities[0].y))
        
        # Extract x and y coordinates
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, marker='o', color=color, linestyle='-', 
                linewidth=2.5, markersize=8, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=color,
                label=f'Vendedor {i+1}')
        
        # Add city numbers
        for j, (x, y) in enumerate(route_coords[1:-1], 1):
            plt.text(x, y, f' {route[j-1]}', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, 
                             edgecolor='none', boxstyle='round,pad=0.2'))
    
    plt.title('Mapa de Rutas de Vendedores', fontsize=14, pad=20)
    plt.xlabel('Coordenada X', fontsize=12)
    plt.ylabel('Coordenada Y', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add watermark
    plt.figtext(0.5, 0.01, 'Generado por Optimización de Rutas', 
               ha='center', fontsize=10, color='gray', alpha=0.7)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Mapa de rutas guardado en: {output_path}")
    return output_path

def export_table_pdf(instances, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Results for Benchmark Instances (BMTSP)", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Generated by LKH + Python", ln=True, align='C')
    pdf.ln(10)

    headers = ["Instance", "k", "m min", "m max", "BKS", "LKH-3", "Gap %", "Time (s)"]
    col_widths = [30, 15, 20, 20, 35, 35, 20, 25]

    pdf.set_font("Arial", 'B', 12)
    for header, w in zip(headers, col_widths):
        pdf.cell(w, 10, header, 1)
    pdf.ln()

    pdf.set_font("Arial", '', 12)
    for inst in instances:
        row = [inst['name'], str(inst['k']), str(inst['mmin']), str(inst['mmax']), f"{inst['bks']:,.2f}", f"{inst['lkh']:,.2f}", f"{inst['gap']:.2f}", f"{inst['time']:.2f}"]
        for datum, w in zip(row, col_widths):
            pdf.cell(w, 10, datum, 1)
        pdf.ln()

    pdf.output(file_path)

def modify_par_file(template_path, output_path, problem_file, solution_file, k, mmin, mmax):
    with open(template_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        for line in lines:
            if line.startswith("PROBLEM_FILE"):
                f.write(f"PROBLEM_FILE = {problem_file}\n")
            elif line.startswith("MTSP_SOLUTION_FILE"):
                f.write(f"MTSP_SOLUTION_FILE = {solution_file}\n")
            elif line.startswith("SALESMEN"):
                f.write(f"SALESMEN = {k}\n")
            elif line.startswith("MTSP_MIN_SIZE"):
                f.write(f"MTSP_MIN_SIZE = {mmin}\n")
            elif line.startswith("MTSP_MAX_SIZE"):
                f.write(f"MTSP_MAX_SIZE = {mmax}\n")
            else:
                f.write(line)

def get_dimension_from_tsp(tsp_path):
    with open(tsp_path, 'r') as f:
        for line in f:
            if line.startswith("DIMENSION"):
                return int(line.strip().split(":")[1])
    raise ValueError(f"No se encontró DIMENSION en {tsp_path}")

def run_gui():
    root = tk.Tk()
    root.withdraw()
    tsp_paths = filedialog.askopenfilenames(title="Selecciona archivos .tsp", filetypes=[("TSP files", "*.tsp")])
    if not tsp_paths:
        print("No se seleccionaron archivos.")
        return

    k = simpledialog.askinteger("Parámetro k", "Número de vendedores (k):", initialvalue=2)
    mmin = simpledialog.askinteger("Parámetro m min", "Tamaño mínimo de ruta (m min):", initialvalue=1)
    mmax = simpledialog.askinteger("Parámetro m max", "Tamaño máximo de ruta (m max):", initialvalue=7)

    run_all_instances_gui(tsp_paths, k, mmin, mmax)

def run_all_instances_gui(tsp_paths, k=2, mmin=1, mmax=7):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    par_template = os.path.join(base_dir, "lkh", "params", "example.par")
    par_tempfile = os.path.join(base_dir, "lkh", "params", "temp.par")
    tour_dir = os.path.join(base_dir, "lkh", "tours")
    maps_dir = os.path.join(base_dir, "lkh", "maps")
    os.makedirs(tour_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    pdf_output = os.path.join(tour_dir, "tabla_resultados.pdf")

    results = []

    bks_dict = {
        "rm_santiago_extended": 215.0,
        "norte_grande": 220.0,
        "patagonia": 252.0,
        "zona_central": 188.0,
        "zona_lacustre": 155.0,
        "zona_sur": 192.0
    }

    for tsp_path in tsp_paths:
        tsp_name = os.path.splitext(os.path.basename(tsp_path))[0]
        tour_file = os.path.join(tour_dir, f"{tsp_name}.tour")

        try:
            dimension = get_dimension_from_tsp(tsp_path)
            if k >= dimension:
                print(f"❌ Error en '{tsp_name}': Número de vendedores (k = {k}) debe ser menor que número de nodos (DIMENSION = {dimension})\n")
                continue

            print(f"✅ Ejecutando {tsp_name}...")
            start_time = time.time()
            modify_par_file(par_template, par_tempfile, tsp_path, tour_file, k, mmin, mmax)
            subprocess.run([r"D:\\descargas\\LKH-3.exe", par_tempfile], check=True)
            duration = time.time() - start_time

            cost, routes = parse_lkh_tour(tour_file)
            
            # Generate and save route map
            cities = load_cities(tsp_path)
            map_filename = f"{tsp_name}_k{k}_mmin{mmin}_mmax{mmax}.png"
            map_path = os.path.join(maps_dir, map_filename)
            plot_route_map(cities, routes, map_path)
            
            bks = bks_dict.get(tsp_name, 0)
            gap = 100 * (cost - bks) / bks if bks > 0 else 0
            results.append({
                "name": tsp_name,
                "k": k,
                "mmin": mmin,
                "mmax": mmax,
                "bks": bks,
                "lkh": cost,
                "gap": gap,
                "time": duration
            })
            print(f"✅ {tsp_name} completado en {duration:.2f} segundos\n")
        except Exception as e:
            print(f"❌ Error en {tsp_name}: {e}\n")

    if results:
        export_table_pdf(results, pdf_output)
        print(f"📄 Resultados exportados a: {pdf_output}")
    else:
        print("⚠️ No se generaron resultados por errores en los parámetros.")

if __name__ == "__main__":
    run_gui()
