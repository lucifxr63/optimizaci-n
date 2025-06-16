import os
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

from heuristica_bmtsp import load_cities, heuristic_bmtsp


def run_ampl(output):
    if not shutil.which("ampl"):
        output.insert(tk.END, "AMPL executable not found in PATH\n")
        return
    cmd = "model ampl/modelo_bmtsp.mod; data ampl/datos_bmtsp.dat; solve;"\
          " display _solve_message;"
    try:
        result = subprocess.run(
            ["ampl"], input=cmd, text=True, capture_output=True, check=True
        )
        output.insert(tk.END, result.stdout + "\n")
    except subprocess.CalledProcessError as e:
        output.insert(tk.END, e.stderr + "\n")


def run_lkh(output):
    if not shutil.which("LKH"):
        output.insert(tk.END, "LKH executable not found in PATH\n")
        return
    try:
        result = subprocess.run(
            ["LKH", "lkh/example.par"], text=True, capture_output=True, check=True
        )
        output.insert(tk.END, result.stdout + "\n")
    except subprocess.CalledProcessError as e:
        output.insert(tk.END, e.stderr + "\n")


def run_heuristic(csv_path, k_entry, max_entry, output):
    try:
        k = int(k_entry.get())
        max_c = int(max_entry.get())
    except ValueError:
        output.insert(tk.END, "Invalid numeric parameters\n")
        return
    if not os.path.exists(csv_path.get()):
        output.insert(tk.END, "CSV file not found\n")
        return
    cities = load_cities(csv_path.get())
    routes, cost = heuristic_bmtsp(cities, k, max_c)
    output.insert(tk.END, f"Total cost: {cost:.2f}\n")
    for i, route in enumerate(routes, 1):
        path = " -> ".join(str(c.idx) for c in route)
        output.insert(tk.END, f"Salesman {i}: {path}\n")
    output.insert(tk.END, "\n")


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
        frm, text="Run AMPL", command=lambda: run_ampl(output)
    ).grid(column=1, row=3, sticky=tk.W)

    ttk.Button(
        frm, text="Run LKH", command=lambda: run_lkh(output)
    ).grid(column=2, row=3, sticky=tk.W)

    root.mainloop()


if __name__ == "__main__":
    main()
