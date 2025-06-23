import os
import subprocess
import math
import csv
import time
import sys
import io
import contextlib
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg for non-interactive plotting
import matplotlib.pyplot as plt
from fpdf import FPDF
import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path

class TextRedirector(io.TextIOBase):
    """
    Una clase para redirigir la salida estándar a un widget de texto de Tkinter.
    """
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag
        self.buffer = ""
        self.lock = False
        
    def write(self, str):
        # Asegurarse de que el texto se escriba en el hilo principal
        self.buffer += str
        if not self.lock:
            self.lock = True
            self.widget.after(100, self._process_buffer)
        return len(str)
    
    def _process_buffer(self):
        if self.buffer:
            self._append_text(self.buffer)
            self.buffer = ""
        self.lock = False
    
    def _append_text(self, text):
        try:
            self.widget.configure(state='normal')
            # Insertar el texto
            self.widget.insert(tk.END, text, (self.tag,))
            # Auto-scroll
            self.widget.see(tk.END)
            # Actualizar la interfaz
            self.widget.update_idletasks()
        except Exception as e:
            print(f"Error al actualizar la consola: {e}")
        finally:
            self.widget.configure(state='disabled')
    
    def flush(self):
        # Procesar cualquier dato restante en el buffer
        if self.buffer:
            self._append_text(self.buffer)
            self.buffer = ""

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

class OptimizacionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimización de Rutas - BMTSP")
        self.root.geometry("800x600")
        
        # Variables
        self.k_var = tk.StringVar(value="2")
        self.mmin_var = tk.StringVar(value="1")
        self.mmax_var = tk.StringVar(value="7")
        self.status_var = tk.StringVar(value="Listo")
        self.selected_files = []
        
        # Configuración de estilos
        self.style = {
            'font': ('Arial', 10),
            'padx': 10,
            'pady': 5,
            'bg': '#f0f0f0'
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.style['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de selección de archivos
        file_frame = tk.LabelFrame(main_frame, text="Archivos de Instancia", font=('Arial', 11, 'bold'), 
                                 bg=self.style['bg'], padx=10, pady=10)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Lista de archivos
        scrollbar = tk.Scrollbar(file_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(file_frame, selectmode=tk.MULTIPLE, width=80, height=15,
                                     yscrollcommand=scrollbar.set, font=self.style['font'])
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Botones de archivo
        btn_frame = tk.Frame(file_frame, bg=self.style['bg'])
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Button(btn_frame, text="Seleccionar Todo", command=self.select_all, 
                 font=self.style['font']).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Desmarcar Todo", command=self.deselect_all, 
                 font=self.style['font']).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Actualizar Lista", command=self.load_instance_files, 
                 font=self.style['font']).pack(side=tk.LEFT, padx=5)
        
        # Frame de parámetros
        param_frame = tk.LabelFrame(main_frame, text="Parámetros de Optimización", 
                                  font=('Arial', 11, 'bold'), bg=self.style['bg'], padx=10, pady=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Controles de parámetros
        tk.Label(param_frame, text="Número de vendedores (k):", bg=self.style['bg'], 
                font=self.style['font']).grid(row=0, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(param_frame, textvariable=self.k_var, width=10, 
                font=self.style['font']).grid(row=0, column=1, sticky='w', pady=5)
        
        tk.Label(param_frame, text="Mínimo ciudades por ruta:", bg=self.style['bg'], 
                font=self.style['font']).grid(row=0, column=2, sticky='e', padx=5, pady=5)
        tk.Entry(param_frame, textvariable=self.mmin_var, width=10, 
                font=self.style['font']).grid(row=0, column=3, sticky='w', pady=5)
        
        tk.Label(param_frame, text="Máximo ciudades por ruta:", bg=self.style['bg'], 
                font=self.style['font']).grid(row=0, column=4, sticky='e', padx=5, pady=5)
        tk.Entry(param_frame, textvariable=self.mmax_var, width=10, 
                font=self.style['font']).grid(row=0, column=5, sticky='w', pady=5)
        
        # Frame de botones de acción
        action_frame = tk.Frame(main_frame, bg=self.style['bg'])
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(action_frame, text="Ejecutar Optimización", command=self.run_optimization,
                 font=('Arial', 11, 'bold'), bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Ver Resultados", command=self.view_results,
                 font=('Arial', 11), bg='#2196F3', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Salir", command=self.root.quit,
                 font=('Arial', 11), bg='#f44336', fg='white').pack(side=tk.RIGHT, padx=5)
        
        # Área de salida de la consola
        console_frame = tk.LabelFrame(main_frame, text="Salida de la consola", 
                                    font=('Arial', 11, 'bold'), bg=self.style['bg'])
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Barra de desplazamiento vertical
        scrollbar = tk.Scrollbar(console_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Widget de texto para la salida de la consola
        self.console_output = tk.Text(console_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                                    bg='#1e1e1e', fg='#00ff00', font=('Consolas', 10), 
                                    insertbackground='white')
        self.console_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.console_output.yview)
        
        # Redirigir la salida estándar al widget de texto
        import sys
        sys.stdout = TextRedirector(self.console_output, "stdout")
        sys.stderr = TextRedirector(self.console_output, "stderr")
        
        # Barra de estado
        status_frame = tk.Frame(main_frame, bg='#e0e0e0', height=25)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, 
                                   bg='#e0e0e0', anchor='w', font=('Arial', 9))
        self.status_label.pack(fill=tk.X, padx=5)
        
        # Cargar archivos de instancia
        self.load_instance_files()
    
    def load_instance_files(self):
        # Obtener el directorio base del proyecto (una carpeta arriba del directorio actual)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        instances_dir = os.path.join(base_dir, "optimizaci-n", "lkh", "instances")
        
        # Asegurarse de que el directorio de instancias exista
        if not os.path.exists(instances_dir):
            os.makedirs(instances_dir)
            self.status_var.set("Carpeta de instancias creada. Por favor, coloque los archivos .tsp en ella.")
            return
            
        # Obtener archivos .tsp
        try:
            tsp_files = [f for f in os.listdir(instances_dir) if f.endswith('.tsp')]
            
            # Actualizar la lista de archivos en la interfaz
            self.file_listbox.delete(0, tk.END)
            for file in sorted(tsp_files):
                self.file_listbox.insert(tk.END, file)
            
            if not tsp_files:
                self.status_var.set("No se encontraron archivos .tsp en la carpeta de instancias.")
            else:
                self.status_var.set(f"Cargados {len(tsp_files)} archivos de instancia.")
                
        except Exception as e:
            self.status_var.set(f"Error al cargar archivos: {str(e)}")
            print(f"Error en load_instance_files: {str(e)}")
    
    def select_all(self):
        self.file_listbox.selection_set(0, tk.END)
    
    def deselect_all(self):
        self.file_listbox.selection_clear(0, tk.END)
    
    def validate_parameters(self):
        try:
            k = int(self.k_var.get())
            mmin = int(self.mmin_var.get())
            mmax = int(self.mmax_var.get())
            
            if k < 1 or mmin < 1 or mmax < 1:
                raise ValueError("Los valores deben ser mayores que cero.")
            if mmin > mmax:
                raise ValueError("El mínimo no puede ser mayor que el máximo.")
                
            return True, k, mmin, mmax
            
        except ValueError as e:
            self.status_var.set(f"Error en parámetros: {str(e)}")
            return False, 0, 0, 0
    
    def get_selected_files(self):
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            self.status_var.set("Error: No se seleccionaron archivos.")
            return []
            
        try:
            # Obtener el directorio base del proyecto (una carpeta arriba del directorio actual)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            instances_dir = os.path.join(base_dir, "optimizaci-n", "lkh", "instances")
            
            # Verificar que el directorio de instancias existe
            if not os.path.exists(instances_dir):
                self.status_var.set(f"Error: No se encontró el directorio de instancias: {instances_dir}")
                return []
                
            # Obtener las rutas completas de los archivos seleccionados
            selected_files = []
            for i in selected_indices:
                filename = self.file_listbox.get(i)
                filepath = os.path.join(instances_dir, filename)
                
                # Verificar que el archivo existe
                if not os.path.exists(filepath):
                    self.status_var.set(f"Advertencia: No se encontró el archivo {filename}")
                    print(f"Archivo no encontrado: {filepath}")
                    continue
                    
                selected_files.append(filepath)
            
            if not selected_files:
                self.status_var.set("Error: Ninguno de los archivos seleccionados existe.")
                
            return selected_files
            
        except Exception as e:
            self.status_var.set(f"Error al obtener archivos: {str(e)}")
            print(f"Error en get_selected_files: {str(e)}")
            return []
    
    def run_optimization(self):
        is_valid, k, mmin, mmax = self.validate_parameters()
        if not is_valid:
            return
            
        selected_files = self.get_selected_files()
        if not selected_files:
            return
            
        # Limpiar la consola antes de una nueva ejecución
        self.console_output.configure(state='normal')
        self.console_output.delete(1.0, tk.END)
        self.console_output.configure(state='disabled')
        
        self.status_var.set("Ejecutando optimización...")
        self.root.update()
        
        # Deshabilitar botones durante la ejecución
        self.toggle_buttons_state('disabled')
        
        # Ejecutar en un hilo separado para no bloquear la interfaz
        import threading
        self.optimization_thread = threading.Thread(
            target=self.run_optimization_thread,
            args=(selected_files, k, mmin, mmax)
        )
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        # Verificar periódicamente si el hilo ha terminado
        self.check_optimization_thread()
    
    def toggle_buttons_state(self, state):
        """Habilitar o deshabilitar botones durante la ejecución"""
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                if widget['text'] not in ['Salir']:  # Mantener el botón Salir habilitado
                    widget['state'] = state
    
    def check_optimization_thread(self):
        """Verificar si el hilo de optimización ha terminado"""
        if hasattr(self, 'optimization_thread') and self.optimization_thread.is_alive():
            # El hilo aún está en ejecución, verificar de nuevo más tarde
            self.root.after(100, self.check_optimization_thread)
        else:
            # El hilo ha terminado, habilitar botones
            self.toggle_buttons_state('normal')
            self.status_var.set("Listo")
    
    def run_optimization_thread(self, selected_files, k, mmin, mmax):
        try:
            self.root.after(0, lambda: self.status_var.set(f"Procesando {len(selected_files)} archivos..."))
            run_all_instances_gui(selected_files, k, mmin, mmax, self.root)
            self.root.after(0, lambda: self.status_var.set("Optimización completada exitosamente."))
            
            # Mostrar mensaje de éxito
            from tkinter import messagebox
            self.root.after(0, lambda: messagebox.showinfo("Éxito", 
                f"Optimización completada.\n\n"
                f"Archivos procesados: {len(selected_files)}\n"
                f"Vendedores: {k}\n"
                f"Rango de ciudades por ruta: {mmin}-{mmax}"))
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            import traceback
            traceback.print_exc()
    
    def view_results(self):
        try:
            # Obtener el directorio base del proyecto (una carpeta arriba del directorio actual)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(base_dir, "optimizaci-n", "lkh", "tours")
            
            # Verificar que el directorio de resultados existe
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
                self.status_var.set("Carpeta de resultados creada. Ejecute la optimización primero.")
                return
                
            pdf_file = os.path.join(results_dir, "tabla_resultados.pdf")
            
            # Verificar que el archivo de resultados existe
            if not os.path.exists(pdf_file):
                self.status_var.set("No se encontró el archivo de resultados. Ejecute la optimización primero.")
                return
                
            # Abrir el archivo PDF con el visor predeterminado
            import webbrowser
            webbrowser.open(pdf_file)
            self.status_var.set(f"Abriendo resultados: {os.path.basename(pdf_file)}")
            
        except Exception as e:
            self.status_var.set(f"Error al abrir resultados: {str(e)}")
            print(f"Error en view_results: {str(e)}")

def run_gui():
    root = tk.Tk()
    app = OptimizacionApp(root)
    
    # Centrar la ventana
    window_width = 900
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    # Evitar redimensionar la ventana
    root.resizable(False, False)
    
    root.mainloop()

def run_all_instances_gui(tsp_paths, k=2, mmin=1, mmax=7, root_window=None):
    # Obtener el directorio base del proyecto (una carpeta arriba del directorio actual)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Ruta al ejecutable LKH-3.exe
    lkh_executable = os.path.join(base_dir, "LKH-3.exe")
    
    # Rutas a los directorios necesarios
    instances_dir = os.path.join(base_dir, "optimizaci-n", "lkh", "instances")
    params_dir = os.path.join(base_dir, "optimizaci-n", "lkh", "params")
    tour_dir = os.path.join(base_dir, "optimizaci-n", "lkh", "tours")
    maps_dir = os.path.join(base_dir, "optimizaci-n", "lkh", "maps")
    
    # Asegurarse de que los directorios existan
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(tour_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    
    # Rutas a los archivos de parámetros
    par_template = os.path.join(params_dir, "example.par")
    par_tempfile = os.path.join(params_dir, "temp.par")
    
    # Ruta al archivo PDF de resultados
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
                print(f"[ERROR] Error en '{tsp_name}': Número de vendedores (k = {k}) debe ser menor que número de nodos (DIMENSION = {dimension})\n")
                continue

            print(f"[OK] Ejecutando {tsp_name}...")
            print(f"Ruta del ejecutable: {lkh_executable}")
            print(f"Ruta del archivo de parámetros: {par_tempfile}")
            
            start_time = time.time()
            modify_par_file(par_template, par_tempfile, tsp_path, tour_file, k, mmin, mmax)
            
            # Verificar que el archivo de parámetros se creó correctamente
            if not os.path.exists(par_tempfile):
                raise FileNotFoundError(f"No se pudo crear el archivo de parámetros: {par_tempfile}")
                
            # Verificar que el ejecutable existe
            if not os.path.exists(lkh_executable):
                raise FileNotFoundError(f"No se encontró el ejecutable LKH-3.exe en: {lkh_executable}")
                
            # Ejecutar LKH-3 y capturar la salida en tiempo real
            try:
                # Configurar el proceso para ejecutarse sin buffer
                process = subprocess.Popen(
                    ['cmd', '/c', lkh_executable, par_tempfile],  # Usar cmd para mejor manejo en Windows
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=0,  # Sin buffer
                    creationflags=subprocess.CREATE_NO_WINDOW  # Evitar ventana de consola
                )
                
                # Leer la salida en tiempo real
                while True:
                    # Leer un carácter a la vez para mejor respuesta
                    output = process.stdout.read(1)
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # Usar print para redirigir a la consola
                        print(output, end='', flush=True)
                        # Forzar la actualización de la interfaz si se proporcionó una ventana raíz
                        if root_window:
                            try:
                                root_window.update_idletasks()
                            except Exception as e:
                                print(f"Error al actualizar la interfaz: {e}")
                
                # Verificar el código de salida
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
                    
            except subprocess.CalledProcessError as e:
                print(f"Error al ejecutar LKH-3: {e}")
                print(f"Comando ejecutado: {lkh_executable} {par_tempfile}")
                raise
                
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
            print(f"[OK] {tsp_name} completado en {duration:.2f} segundos\n")
        except Exception as e:
            print(f"[ERROR] Error en {tsp_name}: {e}\n")

    if results:
        export_table_pdf(results, pdf_output)
        print(f"[INFO] Resultados exportados a: {pdf_output}")
    else:
        print("[WARN] No se generaron resultados por errores en los parámetros.")

if __name__ == "__main__":
    run_gui()
