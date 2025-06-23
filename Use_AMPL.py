import os
import math
import csv
import time
import matplotlib
matplotlib.use('Agg')  # Para usar con Tkinter
import matplotlib.pyplot as plt
import numpy as np
from amplpy import AMPL, add_to_path
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime

# Configurar la ruta de AMPL
AMPL_DIR = r"Z:\DEV\Respos\Universidad\Opti\ampl\AMPL"
if os.path.exists(AMPL_DIR):
    os.environ["PATH"] = AMPL_DIR + os.pathsep + os.environ["PATH"]
    add_to_path(AMPL_DIR)
else:
    raise FileNotFoundError(f"No se encontró el directorio de AMPL en {AMPL_DIR}")

class Ciudad:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = float(x)
        self.y = float(y)

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

def load_cities(csv_path):
    cities = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # saltar encabezado
        for row in reader:
            if len(row) >= 3:
                cities.append(Ciudad(int(row[0]), row[1], row[2]))
    return cities

def reconstruir_ruta(conexiones):
    if not conexiones:
        return [0]
    inicio = 0 if 0 in conexiones else next(iter(conexiones))
    ruta = [inicio]
    actual = inicio
    visitados = set()
    while actual in conexiones and actual not in visitados:
        visitados.add(actual)
        siguiente = conexiones[actual]
        ruta.append(siguiente)
        actual = siguiente
        if siguiente == ruta[0]:
            break
    if 0 in ruta and ruta[0] != 0:
        idx = ruta.index(0)
        ruta = ruta[idx:] + ruta[1:idx+1]
    if ruta[-1] != 0:
        ruta.append(0)
    return ruta

def procesar_rutas(ampl_output):
    import re
    conexiones = {}
    lineas = ampl_output.split('\n')
    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()
        match_vend = re.match(r"(x\s*)?\[\s*(\d+),\*,\*\]", linea)
        if match_vend:
            vendedor = int(match_vend.group(2))
            conexiones[vendedor] = {}
            while i < len(lineas) and ":=" not in lineas[i]:
                i += 1
            i += 1
            while i < len(lineas) and lineas[i].strip() and not lineas[i].strip().startswith('['):
                partes = lineas[i].strip().split()
                if len(partes) > 1:
                    origen = int(partes[0])
                    for j, val in enumerate(partes[1:]):
                        try:
                            if float(val) > 0.5:
                                conexiones[vendedor][origen] = j
                        except ValueError:
                            pass
                i += 1
        else:
            i += 1
    rutas = {}
    for vendedor, links in conexiones.items():
        rutas[vendedor] = reconstruir_ruta(links)
    return rutas

def export_tour_file(rutas, costo_total, output_path):
    """
    Exporta las rutas a un archivo .tour en formato similar a LKH
    
    Args:
        rutas: Diccionario con las rutas por vendedor
        costo_total: Costo total de la solución
        output_path: Ruta del archivo de salida
    """
    with open(output_path, 'w') as f:
        f.write(f"TOUR_SECTION\n")
        f.write(f"Cost: {costo_total:.2f}\n")
        
        for vendedor, ruta in rutas.items():
            # Convertir índices a 1-based para coincidir con LKH
            ruta_str = ' '.join(str(i+1) for i in ruta)
            f.write(f"{ruta_str} -1\n")
        
        f.write("-1\n")  # Fin del tour
        f.write("EOF\n")

def export_routes_csv(rutas, ciudades, output_path):
    """
    Exporta las rutas a un archivo CSV
    
    Args:
        rutas: Diccionario con las rutas por vendedor
        ciudades: Lista de objetos Ciudad
        output_path: Ruta del archivo de salida
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Vendedor', 'Ruta', 'Número de ciudades'])
        
        for vendedor, ruta in rutas.items():
            # Usar solo los índices de las ciudades
            ruta_str = ' -> '.join(f'Ciudad {i}' for i in ruta)
            writer.writerow([f'Vendedor {vendedor}', ruta_str, len(ruta)-1])

def plot_routes_per_salesman(rutas, ciudades, output_path=None):
    """
    Genera un gráfico con las rutas de cada vendedor y lo guarda en un archivo
    
    Args:
        rutas: Diccionario con las rutas por vendedor
        ciudades: Lista de objetos Ciudad
        output_path: Ruta para guardar la imagen (opcional)
    """
    plt.figure(figsize=(12, 10))
    
    # Colores para cada vendedor
    colores = plt.cm.tab10.colors
    
    # Dibujar todas las ciudades
    x = [ciudad.x for ciudad in ciudades]
    y = [ciudad.y for ciudad in ciudades]
    
    # Dibujar el depósito de manera especial
    if ciudades:
        plt.scatter(ciudades[0].x, ciudades[0].y, c='red', s=200, marker='*', 
                   edgecolors='black', zorder=10, label='Depósito')
        plt.scatter(x[1:], y[1:], c='black', s=50, zorder=5, label='Ciudades')
    
    # Dibujar las rutas
    for vendedor, ruta in rutas.items():
        color = colores[(vendedor-1) % len(colores)]
        
        # Conectar los puntos en orden
        ruta_x = [ciudades[i].x for i in ruta]
        ruta_y = [ciudades[i].y for i in ruta]
        
        # Dibujar la ruta
        plt.plot(ruta_x, ruta_y, 'o-', color=color, linewidth=2, markersize=8, 
                label=f'Vendedor {vendedor}')
        
        # Añadir flechas para indicar la dirección
        for i in range(len(ruta_x)-1):
            plt.arrow(ruta_x[i], ruta_y[i], 
                     (ruta_x[i+1]-ruta_x[i])*0.9, (ruta_y[i+1]-ruta_y[i])*0.9,
                     color=color, head_width=0.5, head_length=0.5, 
                     length_includes_head=True, zorder=4)
    
    # Añadir etiquetas a las ciudades
    for i, ciudad in enumerate(ciudades):
        plt.annotate(f"{i}", (ciudad.x + 0.5, ciudad.y + 0.5), fontsize=8)
    
    # Configuraciones del gráfico
    plt.title('Rutas por Vendedor')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal')
    
    # Ajustar márgenes
    plt.tight_layout()
    
    # Guardar o mostrar el gráfico
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_pdf_report(solution, ciudades, output_path, img_path):
    """
    Genera un informe PDF con los resultados de la optimización
    
    Args:
        solution: Diccionario con la solución
        ciudades: Lista de objetos Ciudad
        output_path: Ruta del archivo PDF de salida
        img_path: Ruta de la imagen del mapa de rutas
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    
    # Crear el documento PDF
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Título
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=14,
        alignment=1  # Centrado
    )
    elements.append(Paragraph("Informe de Solución BMTSP", title_style))
    
    # Información general
    elements.append(Paragraph("Información General", styles['Heading2']))
    
    # Crear tabla de información general
    general_data = [
        ["Estado:", solution.get('status', 'Desconocido')],
        ["Costo Total:", f"{solution.get('costo_total', 0):.2f}"],
        ["Número de Vendedores:", len(solution.get('rutas', {}))],
        ["Total de Ciudades:", len(ciudades) - 1],  # -1 por el depósito
        ["Fecha:", time.strftime("%d/%m/%Y %H:%M:%S")]
    ]
    
    # Estilo de la tabla
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ])
    
    # Agregar tabla al documento
    table = Table(general_data, colWidths=[200, 200])
    table.setStyle(table_style)
    elements.append(table)
    elements.append(Spacer(1, 0.25 * inch))
    
    # Detalles de las rutas
    elements.append(Paragraph("Detalles de las Rutas", styles['Heading2']))
    
    # Crear tabla de rutas
    route_data = [["Vendedor", "Ruta", "N° Ciudades", "Distancia"]]
    
    for vendedor, ruta in solution.get('rutas', {}).items():
        # Calcular distancia total de la ruta
        distancia = 0
        for i in range(len(ruta) - 1):
            ciudad_actual = ciudades[ruta[i]]
            ciudad_siguiente = ciudades[ruta[i+1]]
            distancia += ciudad_actual.distance_to(ciudad_siguiente)
        
        # Formatear ruta
        ruta_str = ' → '.join([f"{i}" for i in ruta])
        
        # Agregar fila a la tabla
        route_data.append([
            f"Vendedor {vendedor}",
            ruta_str,
            str(len(ruta) - 1),  # -1 porque el depósito se cuenta dos veces
            f"{distancia:.2f}"
        ])
    
    # Estilo de la tabla de rutas
    route_table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    
    # Ajustar ancho de columnas
    col_widths = [80, 300, 60, 60]
    route_table = Table(route_data, colWidths=col_widths, repeatRows=1)
    route_table.setStyle(route_table_style)
    
    # Agregar tabla de rutas al documento
    elements.append(route_table)
    elements.append(Spacer(1, 0.5 * inch))
    
    # Agregar imagen del mapa
    elements.append(Paragraph("Mapa de Rutas", styles['Heading2']))
    
    # Ajustar tamaño de la imagen
    img = Image(img_path, width=6*inch, height=6*inch)
    elements.append(img)
    
    # Generar el PDF
    doc.build(elements)

def save_solution_files(solution, ciudades, base_filename):
    """
    Guarda todos los archivos de salida para una solución
    
    Args:
        solution: Diccionario con la solución (rutas, costo_total, etc.)
        ciudades: Lista de objetos Ciudad
        base_filename: Nombre base para los archivos de salida (sin extensión)
    """
    # Crear directorio de salida si no existe
    output_dir = os.path.join("optimizaci-n", "ampl", "solutions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar nombres de archivo
    tour_file = os.path.join(output_dir, f"{base_filename}.tour")
    csv_file = os.path.join(output_dir, f"{base_filename}_rutas.csv")
    img_file = os.path.join(output_dir, f"{base_filename}_mapa.png")
    pdf_file = os.path.join(output_dir, f"{base_filename}_informe.pdf")
    
    # Exportar archivos
    export_tour_file(solution['rutas'], solution['costo_total'], tour_file)
    export_routes_csv(solution['rutas'], ciudades, csv_file)
    plot_routes_per_salesman(solution['rutas'], ciudades, img_file)
    
    # Generar informe PDF
    generate_pdf_report(solution, ciudades, pdf_file, img_file)
    
    return {
        'tour_file': tour_file,
        'csv_file': csv_file,
        'img_file': img_file,
        'pdf_file': pdf_file
    }

def run_ampl_optimization(ciudades, k=2, max_ciudades=5, callback=None):
    """
    Ejecuta la optimización usando AMPL
    
    Args:
        ciudades: Lista de objetos Ciudad
        k: Número de vendedores
        max_ciudades: Máximo de ciudades por vendedor
        callback: Función para mostrar progreso (opcional)
        
    Returns:
        dict: Rutas por vendedor y costo total
    """
    if callback:
        callback("Iniciando optimización con AMPL...")
    
    try:
        # Inicializar AMPL
        ampl = AMPL()
        
        # Cargar el modelo
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "ampl", "modelo_bmtsp.mod")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_path}")
        
        if callback:
            callback(f"Cargando modelo desde: {model_path}")
        
        ampl.read(model_path)
        
        # Establecer parámetros
        # n = número de ciudades sin contar el depósito (el modelo espera esto)
        n = len(ciudades) - 1  # Asumiendo que la ciudad 0 es el depósito
        ampl.param["n"] = n
        ampl.param["k"] = k
        ampl.param["max_ciudades"] = max_ciudades
        
        if callback:
            callback(f"Parámetros: n={n} (ciudades), k={k} (vendedores), max_ciudades={max_ciudades}")
        
        # Calcular matriz de distancias
        if callback:
            callback("Calculando matriz de distancias...")
        
        # Crear matriz de distancias con índices que coincidan con el modelo AMPL
        # El modelo espera índices de 0 a n, donde 0 es el depósito
        n = len(ciudades) - 1  # Número de ciudades sin contar el depósito
        dist = {}
        
        # Llenar la matriz de distancias
        for i in range(n + 1):  # Incluye el depósito (0) y las ciudades (1..n)
            for j in range(n + 1):
                dist[(i, j)] = int(ciudades[i].distance_to(ciudades[j]))
        
        if callback:
            callback(f"Dimensiones de la matriz de distancias: {n+1}x{n+1} (incluyendo depósito)")
            callback(f"Número de ciudades (sin depósito): {n}")
            
        # Cargar datos en AMPL
        try:
            ampl.param["dist"] = dist
        except Exception as e:
            # Mostrar información de depuración
            if callback:
                callback(f"Error al cargar distancias: {str(e)}")
                callback(f"Rango de índices: {min(k[0] for k in dist.keys())}-{max(k[0] for k in dist.keys())} para filas, {min(k[1] for k in dist.keys())}-{max(k[1] for k in dist.keys())} para columnas")
            raise
        
        # Configurar solver
        ampl.option["solver"] = "highs"  # Usar HiGHS como solver (gratuito)
        
        # Resolver el modelo
        if callback:
            callback("Resolviendo el modelo...")
        
        ampl.solve()
        
        # Procesar resultados
        if callback:
            callback("Procesando resultados...")
        
        # Obtener las variables de decisión x
        x = ampl.get_variable("x")
        
        # Obtener los valores de las variables
        rutas = {}
        
        # Iterar sobre todas las variables x[s,i,j]
        for s in range(1, k + 1):  # Vendedores
            rutas[s] = {}
            for i in range(len(ciudades)):  # Nodos origen
                for j in range(len(ciudades)):  # Nodos destino
                    if i != j:  # No hay arcos de un nodo a sí mismo
                        val = x.get((s, i, j)).value()
                        if val > 0.5:  # Considerar arcos seleccionados
                            rutas[s][i] = j
        
        if callback:
            callback(f"Arcos seleccionados: {sum(len(arcs) for arcs in rutas.values())}")
            
            # Mostrar información de depuración
            for s in rutas:
                callback(f"Vendedor {s} tiene {len(rutas[s])} arcos")
                for i, j in rutas[s].items():
                    callback(f"  {i} -> {j}")
        
        # Reconstruir las rutas completas
        rutas_completas = {}
        for v, conexiones in rutas.items():
            rutas_completas[v] = reconstruir_ruta(conexiones)
        
        # Obtener el costo total
        total_distance = ampl.get_objective("Total_Distance").value()
        
        if callback:
            callback("Optimización completada exitosamente")
        
        return {
            'rutas': rutas_completas,
            'costo_total': total_distance,
            'status': ampl.solve_result
        }
        
    except Exception as e:
        error_msg = f"Error en la optimización: {str(e)}"
        if callback:
            callback(error_msg)
        raise RuntimeError(error_msg) from e

def select_dat_file():
    """
    Muestra un diálogo para seleccionar un archivo .dat de la carpeta ampl/datos/
    
    Returns:
        str: Ruta completa al archivo .dat seleccionado o None si se cancela
    """
    import tkinter as tk
    from tkinter import filedialog, ttk
    
    # Obtener la lista de archivos .dat
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "ampl", "datos")
    
    if not os.path.exists(datos_dir):
        print(f"No se encontró el directorio de datos: {datos_dir}")
        return None
    
    # Buscar archivos .dat
    dat_files = [f for f in os.listdir(datos_dir) if f.endswith('.dat')]
    
    if not dat_files:
        print(f"No se encontraron archivos .dat en {datos_dir}")
        return None
    
    # Crear ventana de selección
    root = tk.Tk()
    root.title("Seleccionar archivo de datos")
    root.geometry("500x300")
    
    # Estilo
    style = ttk.Style()
    style.configure('TButton', padding=5, font=('Arial', 10))
    style.configure('TLabel', font=('Arial', 10))
    
    # Variable para el resultado
    selected_file = None
    
    def on_select():
        nonlocal selected_file
        selection = listbox.curselection()
        if selection:
            selected_file = os.path.join(datos_dir, listbox.get(selection[0]))
            root.destroy()
    
    def on_cancel():
        root.destroy()
    
    # Título
    ttk.Label(root, text="Seleccione un archivo de datos (.dat)", 
             font=('Arial', 12, 'bold')).pack(pady=10)
    
    # Lista de archivos
    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    scrollbar = ttk.Scrollbar(frame)
    scrollbar.pack(side='right', fill='y')
    
    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=('Arial', 10))
    for file in sorted(dat_files):
        listbox.insert('end', file)
    listbox.pack(side='left', fill='both', expand=True)
    listbox.selection_set(0)  # Seleccionar primer elemento por defecto
    
    scrollbar.config(command=listbox.yview)
    
    # Botones
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)
    
    ttk.Button(btn_frame, text="Seleccionar", command=on_select).pack(side='left', padx=5)
    ttk.Button(btn_frame, text="Cancelar", command=on_cancel).pack(side='left', padx=5)
    
    # Centrar ventana
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Mostrar ventana
    root.mainloop()
    
    return selected_file

def test_ampl():
    """Función principal para probar la optimización con AMPL"""
    # Seleccionar archivo .dat
    dat_path = select_dat_file()
    if not dat_path:
        print("No se seleccionó ningún archivo. Saliendo...")
        return
    
    print(f"\nArchivo seleccionado: {os.path.basename(dat_path)}")
    
    try:
        # Leer el archivo .dat
        with open(dat_path, 'r') as f:
            content = f.read()
        
        # Inicializar parámetros con valores por defecto
        n = 0
        k = 2
        max_ciudades = 5
        dist_matrix = {}
        
        # Procesar el contenido del archivo
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Buscar parámetros
            if line.startswith('param n :='):
                n = int(line.split(':=')[1].replace(';', '').strip())
            elif line.startswith('param k :='):
                k = int(line.split(':=')[1].replace(';', '').strip())
            elif line.startswith('param max_ciudades :='):
                max_ciudades = int(line.split(':=')[1].replace(';', '').strip())
            
            # Buscar matriz de distancias
            elif line.startswith('param dist :'):
                # Obtener índices de ciudades de la primera línea
                header_parts = line.split(':')
                if len(header_parts) > 1:
                    # Obtener la parte después de 'param dist :' y eliminar espacios
                    indices_line = header_parts[1].strip()
                    # Extraer los índices de las ciudades
                    city_indices = [int(x) for x in indices_line.split() if x.isdigit()]
                    
                    # Procesar las filas de la matriz de distancias
                    i += 1  # Mover a la siguiente línea
                    while i < len(lines) and not lines[i].startswith(';'):
                        parts = lines[i].split()
                        if parts and parts[0].isdigit():
                            city_from = int(parts[0])
                            dist_matrix[city_from] = {}
                            for j in range(1, len(parts)):
                                if j-1 < len(city_indices):
                                    city_to = city_indices[j-1]
                                    try:
                                        dist_matrix[city_from][city_to] = float(parts[j])
                                    except (ValueError, IndexError):
                                        pass
                        i += 1
                    break  # Salir del bucle una vez procesada la matriz
            i += 1
        
        # Verificar que se cargó la matriz de distancias
        if not dist_matrix:
            print("Error: No se pudo cargar la matriz de distancias del archivo.")
            return
        
        # Crear ciudades ficticias basadas en la matriz de distancias
        # Como no tenemos coordenadas, usaremos índices y distancias
        ciudades = [Ciudad(i, i*10, 0) for i in range(n+1)]  # +1 porque incluye el depósito
        
        print(f"\nDatos cargados exitosamente:")
        print(f"- Número de ciudades: {n}")
        print(f"- Número de vendedores (k): {k}")
        print(f"- Máximo de ciudades por vendedor: {max_ciudades}")
        
        # Preguntar si desea modificar los parámetros
        try:
            modificar = input("\n¿Desea modificar los parámetros? (s/n): ").strip().lower()
            if modificar == 's':
                try:
                    k = int(input(f"Número de vendedores [{k}]: ") or k)
                    max_ciudades = int(input(f"Máximo de ciudades por vendedor [{max_ciudades}]: ") or max_ciudades)
                except ValueError:
                    print("Valor inválido, usando valores por defecto.")
        except KeyboardInterrupt:
            print("\nOperación cancelada por el usuario.")
            return
    
        # Ejecutar la optimización
        print("\nIniciando optimización...")
        try:
            # Crear ciudades con coordenadas más realistas
            # Usamos una semilla fija para resultados consistentes
            np.random.seed(42)
            
            # Generar coordenadas aleatorias dentro de un área cuadrada
            scale = 100  # Escala para las coordenadas
            x_coords = np.random.uniform(0, scale, n+1)
            y_coords = np.random.uniform(0, scale, n+1)
            
            # Asegurarnos de que el depósito (índice 0) esté en el centro
            x_coords[0] = scale / 2
            y_coords[0] = scale / 2
            
            # Crear las ciudades con estas coordenadas
            ciudades = [Ciudad(i, x, y) for i, (x, y) in enumerate(zip(x_coords, y_coords))]
            
            # Ejecutar la optimización real
            resultado = run_ampl_optimization(ciudades, k, max_ciudades)
            
            if resultado:
                print("\nOptimización completada exitosamente")
                print(f"Costo total: {resultado.get('costo_total', 'N/A')}")
                print(f"Estado: {resultado.get('status', 'N/A')}")
                
                # Mostrar rutas por vendedor
                if 'rutas' in resultado:
                    print("\nRutas por vendedor:")
                    for vendedor, ruta in resultado['rutas'].items():
                        print(f"{vendedor}: {' -> '.join(map(str, ruta))}")
                        
                        # Si las ciudades tienen nombres, mostrarlos también
                        if ciudades and hasattr(ciudades[0], 'nombre'):
                            print(f"   ({' -> '.join(ciudades[i].nombre for i in ruta if i < len(ciudades))})")
                
                # Guardar archivos de salida
                try:
                    output_files = save_solution_files(
                        solution=resultado,
                        ciudades=ciudades,
                        base_filename=f"solucion_ampl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    
                    print("\nArchivos generados:")
                    for tipo, archivo in output_files.items():
                        print(f"- {tipo}: {os.path.abspath(archivo)}")
                        
                except Exception as e:
                    print(f"\nAdvertencia: No se pudieron guardar todos los archivos: {e}")
                    
            else:
                print("\nNo se obtuvo ningún resultado de la optimización")
                
        except Exception as e:
            print(f"\nError durante la optimización: {e}")
            import traceback
            traceback.print_exc()

        input("\nPresione Enter para salir...")

    except Exception as e:
        print(f"\nError al procesar el archivo: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPresione Enter para salir...")

if __name__ == "__main__":
    test_ampl()
