# Bounded Multiple Traveling Salesman Problem (BMTSP)

This repository contains three complementary approaches to solve or approximate
instances of the BMTSP where multiple salesmen must visit all cities while
respecting a limit on the number of cities per salesman.

```
ampl/   - AMPL model and example data
python/ - Python heuristic and helper scripts
lkh/    - Example input and parameter files for LKH 3.0
```

## 1. AMPL Model

The AMPL model (`ampl/modelo_bmtsp.mod`) implements the BMTSP with MTZ
subtour elimination. Example data for ten cities and two salesmen is provided
in `ampl/datos_bmtsp.dat`.

Run in AMPL (assuming `ampl` command is available):

```ampl
model ampl/modelo_bmtsp.mod;
data ampl/datos_bmtsp.dat;
solve;
```

The model minimizes total distance traveled while ensuring that each salesman
starts and ends at the depot and visits at most `max_ciudades` cities.

## 2. Python Heuristic

`python/heuristica_bmtsp.py` implements a simple heuristic:

1. Cities (except depot) are randomly assigned to salesmen respecting
   `max_ciudades`.
2. Each salesman route is built using the nearest neighbor algorithm and
   improved with a basic 2‑opt routine.

Example usage with the provided coordinates in `python/ciudades.csv`:

```bash
python python/heuristica_bmtsp.py python/ciudades.csv 2 5
```

If `matplotlib` is installed, a plot of the computed routes will be shown.

A helper script `python/run_all.py` can run the heuristic and, if the `LKH`
executable is installed, call it as well:

```bash
python python/run_all.py python/ciudades.csv 2 5 --lkh lkh/example.tsp lkh/example.par
```

## 3. LKH 3.0 Configuration

The `lkh` directory contains an example TSPLIB file (`example.tsp`) and a
parameter file (`example.par`). After compiling LKH and ensuring the `LKH`
executable is in your `PATH`, run:

```bash
cd lkh
LKH example.par
```

The resulting tour will be written to `example.tour` and can be compared with
the heuristic output.

## Requirements

- Python 3.8+ (the heuristic uses only the standard library; `matplotlib` is
  optional for plotting)
- AMPL for solving the model
- LKH 3.0 (optional) for the exact solver

## File Overview

- `ampl/modelo_bmtsp.mod` – AMPL formulation
- `ampl/datos_bmtsp.dat` – example data
- `python/heuristica_bmtsp.py` – heuristic implementation
- `python/run_all.py` – runner script
- `python/ciudades.csv` – sample coordinates
- `lkh/example.tsp` and `lkh/example.par` – input for LKH
- `lkh/README.md` – brief instructions for running LKH

With these resources you can test small instances, compare heuristic results
with those from LKH, and adapt the AMPL model for larger problems.

## 4. Simple GUI

The script `python/gui.py` provides a tiny Tkinter interface to run the
heuristic, the AMPL model or LKH from one place. Launch it with:

```bash
python python/gui.py
```

Enter the CSV file and parameters for the heuristic (defaults are already
filled) and then click **Run Heuristic**, **Run AMPL** or **Run LKH**. Output
from the selected solver will appear in the text box. AMPL or LKH must be
available in your `PATH` for those buttons to work.
