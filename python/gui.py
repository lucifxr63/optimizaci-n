def parse_ampl_routes(ampl_text, cities):
    """Parse salesman routes from AMPL ``display x`` output.

    AMPL may print ``x`` either as individual assignments ``x[s,i,j] = 1`` or as
    matrices.  This parser supports both formats.
    """

    edges = []

    # First, look for ``x[s,i,j] = 1`` lines
    pattern = re.compile(r"x\[(\d+),(\d+),(\d+)\]\s*=\s*1")
    for s, i, j in pattern.findall(ampl_text):
        edges.append((int(s), int(i), int(j)))

    # If nothing found, attempt to parse matrix form "x [s,*,*]" printed by AMPL
    if not edges:
        lines = iter(ampl_text.splitlines())
        for line in lines:
            m = re.match(r"x\s*\[(\d+),\*,\*\]", line.strip())
            if not m:
                continue
            s = int(m.group(1))

            # Read header with column indices
            header = next(lines, "")
            cols = [int(n) for n in re.findall(r"-?\d+", header)]

            for row in lines:
                row = row.strip()
                if not row or row.startswith("[") or row.startswith(";"):
                    break
                tokens = re.findall(r"-?\d+", row)
                if len(tokens) != len(cols) + 1:
                    continue
                i = int(tokens[0])
                vals = tokens[1:]
                for col, val in zip(cols, vals):
                    if val == "1":
                        edges.append((s, i, col))


def parse_ampl_routes(ampl_text, cities):
    """Parse salesman routes from AMPL ``display x`` output.

    AMPL may print ``x`` either as individual assignments ``x[s,i,j] = 1`` or as
    matrices.  This parser supports both formats and handles multiple salesmen.
    """

    edges = []

    # 1) ``x[s,i,j] = 1`` assignments
    pattern = re.compile(r"x\[(\d+),(\d+),(\d+)\]\s*=\s*1")
    for s, i, j in pattern.findall(ampl_text):
        edges.append((int(s), int(i), int(j)))

    # 2) Matrix form ``x [s,*,*]`` possibly written as ``[s,*,*]``
    matrix_pat = re.compile(r"(?:x\s*)?\[(\d+),\*,\*\]")
    lines = ampl_text.splitlines()
    idx = 0
    while idx < len(lines):
        m = matrix_pat.match(lines[idx].strip())
        if not m:
            idx += 1
            continue
        s = int(m.group(1))
        idx += 1
        if idx >= len(lines):
            break
        header = lines[idx]
        cols = [int(n) for n in re.findall(r"-?\d+", header)]
        idx += 1
        while idx < len(lines):
            row = lines[idx].strip()
            if not row or row.startswith("[") or row.startswith(";"):
                break
            tokens = re.findall(r"-?\d+", row)
            if len(tokens) == len(cols) + 1:
                i = int(tokens[0])
                for col, val in zip(cols, tokens[1:]):
                    if val == "1":
                        edges.append((s, i, col))
            idx += 1
        # do not consume delimiter line; outer loop will reconsider it

    if not edges:
        return []

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

            if nxt == 0:
                route.append(city_map[0])
                break

            route.append(city_map[nxt])
            current = nxt

        routes.append(route)

    return routes


            if current == 0 or len(visited) > len(city_map):
                break

        if route[-1].idx != 0:
            route.append(city_map[0])

        routes.append(route)


def run_ampl(output, csv_path):
    """Execute the AMPL model and display the resulting routes."""
    # Allow configuration via the AMPL_PATH environment variable.  If not
    # provided, fall back to searching ``ampl`` in the PATH.
    ampl_path = os.environ.get("AMPL_PATH", "ampl")

    # Verificar si el archivo ejecutable existe o está en PATH
    if not shutil.which(ampl_path):
        append_output(
            output,
            f"Error: No se encontró el ejecutable de AMPL ({ampl_path}).\n",
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
    import tempfile
    import os
    
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
                    path = " → ".join(str(c.idx) for c in route)
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

    return routes

