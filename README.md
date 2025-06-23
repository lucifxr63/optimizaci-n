# Problema Acotado del Vendedor Viajero Múltiple (BMTSP)

Este repositorio contiene múltiples enfoques para resolver instancias del BMTSP, donde múltiples vendedores deben visitar todas las ciudades respetando un límite en el número de ciudades por vendedor.

## Estructura del Proyecto

```
optimizaci-n/
├── ampl/               # Modelo AMPL y datos de ejemplo
├── python/             # Scripts de Python y heurísticas
├── lkh/                # Archivos de entrada para LKH 3.0
└── jupiter/            # Notebooks de Jupyter con análisis
```

## 1. Modelo AMPL (`ampl/modelo_bmtsp.mod`)

Implementación del BMTSP con eliminación de subtours MTZ. El modelo minimiza la distancia total recorrida asegurando que cada vendedor comience y termine en el depósito y visite como máximo `max_ciudades` ciudades.

### Funciones Principales:
- **Distancia_Total**: Función objetivo que minimiza la distancia total recorrida
- **Restricciones**:
  - `UnaEntrada` y `UnaSalida`: Cada ciudad tiene exactamente una entrada y una salida
  - `Flujo`: Conservación de flujo para cada vendedor
  - `InicioDeposito` y `FinDeposito`: Cada vendedor comienza y termina en el depósito
  - `LimiteCiudades`: Límite de ciudades por vendedor
  - `MTZ`: Eliminación de subtours usando variables MTZ

## 2. Aplicación Python (`Use_AMPL.py` y `Use_Chile.py`)

### Clases Principales:

#### `Ciudad`
- Almacena información de cada ciudad (índice, coordenadas x, y)
- Método `distance_to`: Calcula la distancia euclidiana a otra ciudad

#### `OptimizacionApp` (Use_Chile.py)
Interfaz gráfica para la optimización de rutas:
- Carga instancias de problemas
- Configura parámetros (número de vendedores, ciudades máximas)
- Ejecuta diferentes métodos de optimización
- Visualiza resultados

### Funciones Principales:

#### En `Use_AMPL.py`:
- `run_ampl_optimization`: Ejecuta la optimización usando AMPL
- `reconstruir_ruta`: Reconstruye la ruta completa a partir de las conexiones
- `procesar_rutas`: Procesa la salida de AMPL para extraer las rutas
- `generate_pdf_report`: Genera un informe PDF con los resultados
- `plot_routes_per_salesman`: Grafica las rutas de cada vendedor
- `save_solution_files`: Guarda los resultados en diferentes formatos

#### En `Use_Chile.py`:
- `setup_ui`: Configura la interfaz gráfica
- `load_instance_files`: Carga los archivos de instancia disponibles
- `run_optimization`: Ejecuta la optimización seleccionada
- `update_status`: Actualiza la barra de estado
- `show_help`: Muestra la ayuda de la aplicación

## 3. Heurística Python (`python/heuristica_bmtsp.py`)

Implementa una heurística que:
1. Asigna ciudades aleatoriamente a vendedores respetando `max_ciudades`
2. Construye rutas usando el algoritmo del vecino más cercano
3. Mejora las rutas con 2-opt

### Funciones Principales:
- `load_cities`: Carga ciudades desde archivo CSV
- `nearest_neighbor_route`: Construye una ruta usando el vecino más cercano
- `two_opt`: Mejora la ruta con el algoritmo 2-opt
- `heuristic_bmtsp`: Función principal que orquesta la heurística


## Requerimientos

- Python 3.8+
- Bibliotecas Python:
  - `matplotlib` para visualización
  - `numpy` para cálculos numéricos
  - `reportlab` para generación de PDFs
  - `tkinter` para la interfaz gráfica
- AMPL (para el modelo de optimización)
- LKH 3.0 (opcional, para solución exacta)

## Uso

1. **Interfaz Gráfica**:
   ```bash
   python Use_Chile.py
   ```

2. **Línea de Comandos (AMPL)**:
   ```bash
   python Use_AMPL.py
   ```

3. **Heurística Directa**:
   ```bash
   python python/heuristica_bmtsp.py datos/ciudades.csv 2 5
   ```

## Estructura de Archivos

- `ampl/`
  - `modelo_bmtsp.mod`: Modelo de optimización AMPL
  - `datos/`: Archivos de datos para diferentes instancias
  - `solutions/`: Soluciones generadas
- `jupiter/`
  - `1.ipynb`: Implementación de la heurística
- `lkh/`: Configuración para el solver LKH
- `jupiter/`: Notebooks de análisis y desarrollo

## Notas

- El código está documentado en español
- Se incluyen ejemplos para diferentes tamaños de problemas
- La interfaz gráfica permite una interacción sencilla con los diferentes métodos de solución
