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
    
    # Obtener la ruta base del proyecto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tsp_file = os.path.join(base_dir, "ampl", "bmtsp.tsp")
    par_file = os.path.join(base_dir, "ampl", "bmtsp.par")
    
    # Verificar que los archivos existan
    if not os.path.exists(tsp_file):
        append_output(output, f"Error: No se encontró el archivo TSP en {tsp_file}\n")
        return
    if not os.path.exists(par_file):
        append_output(output, f"Error: No se encontró el archivo de parámetros en {par_file}\n")
        return
    
    try:
        append_output(output, "Ejecutando LKH...\n")
        result = subprocess.run(
            ["LKH", par_file],
            text=True,
            capture_output=True,
            check=True
        )
        
        # Mostrar resultados
        append_output(output, "\n=== RESULTADOS DE LKH ===\n")
        append_output(output, result.stdout)
        
        if result.stderr:
            append_output(output, "\n=== ADVERTENCIAS ===\n")
            append_output(output, result.stderr)
        
        # Procesar archivo de tour
        tour_file = os.path.splitext(par_file)[0] + ".tour"
        if os.path.exists(tour_file):
            with open(tour_file) as f:
                tour_content = f.read()
                append_output(output, "\n=== TOUR ENCONTRADO ===\n")
                append_output(output, tour_content)
                
                # Cargar las ciudades y mostrar las rutas
                try:
                    cities = load_cities(csv_path.get())
                    routes = parse_lkh_tour(tour_file, cities)
                    if routes:
                        append_output(output, "\nRutas encontradas:\n")
                        for i, route in enumerate(routes, 1):
                            path = " → ".join(str(c.idx) for c in route)
                            append_output(output, f"Vendedor {i}: {path}\n")
                        plot_routes(routes)
                except Exception as e:
                    append_output(output, f"\nError al mostrar las rutas: {str(e)}\n")
        else:
            append_output(output, "\nNo se encontró el archivo de tour generado\n")
            
    except subprocess.CalledProcessError as e:
        error_msg = "\n=== ERROR AL EJECUTAR LKH ===\n"
        error_msg += f"Código de salida: {e.returncode}\n"
        if e.stdout:
            error_msg += f"Salida estándar:\n{e.stdout}\n"
        if e.stderr:
            error_msg += f"Error estándar:\n{e.stderr}\n"
        append_output(output, error_msg)
    except Exception as e:
        append_output(output, f"\nError inesperado: {str(e)}\n")


def select_file(entry_widget):
    """Abre un diálogo para seleccionar un archivo CSV."""
    from tkinter import filedialog
    filename = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if filename:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, filename)


def create_gui():
    """Crea la interfaz gráfica principal."""
    root = tk.Tk()
    root.title("Optimización BMTSP")
    
    # Configuración del estilo
    style = ttk.Style()
    style.configure("TButton", padding=6, relief="flat", background="#ccc")
    style.configure("TFrame", background="#f0f0f0")
    style.configure("TLabel", background="#f0f0f0")
    
    # Frame principal
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Frame para la entrada de archivo
    file_frame = ttk.Frame(main_frame)
    file_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(file_frame, text="Archivo CSV:").pack(side=tk.LEFT, padx=(0, 5))
    
    csv_path = tk.StringVar()
    file_entry = ttk.Entry(file_frame, textvariable=csv_path, width=50)
    file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    browse_btn = ttk.Button(file_frame, text="Examinar...", 
                          command=lambda: select_file(file_entry))
    browse_btn.pack(side=tk.LEFT, padx=(5, 0))
    
    # Frame para los botones
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    
    # Botones
    ampl_btn = ttk.Button(button_frame, text="Ejecutar AMPL", 
                         command=lambda: run_ampl(output, csv_path))
    ampl_btn.pack(side=tk.LEFT, padx=5)
    
    lkh_btn = ttk.Button(button_frame, text="Ejecutar LKH", 
                        command=lambda: run_lkh(output, csv_path))
    lkh_btn.pack(side=tk.LEFT, padx=5)
    
    # Área de salida
    output_frame = ttk.LabelFrame(main_frame, text="Salida", padding="5")
    output_frame.pack(fill=tk.BOTH, expand=True)
    
    output = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        width=80,
        height=20,
        font=('Consolas', 10)
    )
    output.pack(fill=tk.BOTH, expand=True)
    
    # Barra de estado
    status_bar = ttk.Label(
        main_frame,
        text="Listo",
        relief=tk.SUNKEN,
        anchor=tk.W
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Redirigir stdout y stderr a la salida
    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_widget = text_widget
        
        def write(self, message):
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)
        
        def flush(self):
            pass
    
    sys.stdout = StdoutRedirector(output)
    sys.stderr = StdoutRedirector(output)
    
    return root


if __name__ == "__main__":
    app = create_gui()
    app.mainloop()