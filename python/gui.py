import os
import re
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from dataclasses import dataclass

from heuristica_bmtsp import load_cities, heuristic_bmtsp, plot_routes


def append_output(widget, text):
    widget.insert(tk.END, text)
    widget.see(tk.END)


def parse_ampl_routes(ampl_text, cities):
    """Parse routes from AMPL output displaying x[s,i,j] variables."""
    # Primero intentar con el formato de tabla
    table_pattern = re.compile(r'\[([12]),\*,\*\]\s*:.*?:=(.*?)(?=\n\s*\[|\Z)', re.DOTALL)
    tables = table_pattern.findall(ampl_text)
    
    if not tables:
        # Si no encontramos tablas, intentar con el formato anterior
        pattern = re.compile(r"x\[(\d+),(\d+),(\d+)\]\s*=\s*1")
        edges = pattern.findall(ampl_text)
        if not edges:
            return []
        edges = [(int(s), int(i), int(j)) for s, i, j in edges]
    else:
        # Procesar las tablas
        edges = []
        for s, table in tables:
            s = int(s)
            # Procesar cada línea de la tabla
            for line in table.split('\n'):
                if not line.strip() or ':' not in line:
                    continue
                # Extraer la fila (i) y los valores
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                i = int(parts[0].strip())
                values = parts[1].split()
                # Los índices de columna van de 0 a len(values)-1
                for j, val in enumerate(values):
                    try:
                        if int(val) == 1 and i != j:  # Evitar bucles a sí mismo
                            edges.append((s, i, j))
                    except ValueError:
                        continue
    
    if not edges:
        return []
        
    # Construir las rutas a partir de las aristas
    salesmen = sorted({s for s, _, _ in edges})
    by_s = {s: {} for s in salesmen}
    for s, i, j in edges:
        by_s[s][i] = j
    
    city_map = {c.idx: c for c in cities}
    routes = []
    
    for s in salesmen:
        if s not in by_s:
            continue
        route = [city_map[0]]  # Empezar en el depósito
        current = 0
        visited = set()
        
        while True:
            if current in visited:
                break  # Evitar ciclos infinitos
            visited.add(current)
            
            nxt = by_s[s].get(current)
            if nxt is None or nxt == current:
                break
                
            if nxt == 0:  # Si volvemos al depósito, terminar la ruta
                route.append(city_map[0])
                break
                
            route.append(city_map[nxt])
            current = nxt
            
            # Si hemos vuelto al depósito o no hay más ciudades que visitar
            if current == 0 or len(visited) > len(city_map):
                break
        
        # Asegurarse de terminar en el depósito
        if route[-1].idx != 0:
            route.append(city_map[0])
            
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
    """Execute the AMPL model and display the resulting routes."""
    # Usar la ruta completa al ejecutable de AMPL
    ampl_path = r"D:\DEV\AMPL\ampl.exe"

    # Verificar si el archivo ejecutable existe
    if not os.path.exists(ampl_path):
        append_output(
            output,
            f"Error: No se encontró el ejecutable de AMPL en {ampl_path}.\n"
            "Por favor, verifica que la ruta sea correcta.\n"
        )
        return
    
    # Obtener la ruta base del proyecto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "ampl", "modelo_bmtsp.mod")
    data_path = os.path.join(base_dir, "ampl", "datos_bmtsp.dat")
    
    # Verificar que los archivos existan
    if not os.path.exists(model_path):
        append_output(output, f"Error: No se encontró el archivo del modelo en {model_path}\n")
        return
    if not os.path.exists(data_path):
        append_output(output, f"Error: No se encontró el archivo de datos en {data_path}\n")
        return
    
    # Crear un archivo temporal para los comandos de AMPL
    
    # Crear un archivo temporal
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.run', delete=False) as f:
            temp_file = f.name
            f.write(f'''
            model "{model_path}";
            data "{data_path}";
            option solver cplex;
            solve;
            
            display Total_Distance;
            display x;
            ''')
        
        # Usar el archivo temporal con AMPL
        cmd = [ampl_path, temp_file]
        
        append_output(output, "Ejecutando AMPL...\n")
        
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True
        )
        
        # Mostrar resultados crudos
        append_output(output, "\n=== RESULTADOS DE AMPL ===\n")
        append_output(output, result.stdout)

        if result.stderr:
            append_output(output, "\n=== ADVERTENCIAS ===\n")
            append_output(output, result.stderr)

        # Cargar las ciudades y mostrar las rutas
        try:
            cities = load_cities(csv_path.get())
            routes = parse_ampl_routes(result.stdout, cities)
            if routes:
                append_output(output, "\nRutas encontradas:\n")
                for i, route in enumerate(routes, 1):
                    path = " -> ".join(str(c.idx) for c in route)
                    append_output(output, f"Vendedor {i}: {path}\n")
                plot_routes(routes)
            else:
                append_output(output, "No se pudieron interpretar rutas en la salida de AMPL\n")
        except Exception as e:
            append_output(output, f"\nError al mostrar las rutas: {str(e)}\n")
            
    except subprocess.CalledProcessError as e:
        error_msg = f"\n=== ERROR AL EJECUTAR AMPL ===\n"
        error_msg += f"Código de salida: {e.returncode}\n"
        if e.stdout:
            error_msg += f"Salida estándar:\n{e.stdout}\n"
        if e.stderr:
            error_msg += f"Error estándar:\n{e.stderr}\n"
        append_output(output, error_msg)
    except Exception as e:
        append_output(output, f"\nError inesperado: {str(e)}\n")
    finally:
        # Asegurarse de eliminar el archivo temporal
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


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
