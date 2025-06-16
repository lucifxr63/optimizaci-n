import os
import subprocess

def test_ampl():
    # Ruta base del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas a los archivos
    model_path = os.path.join(base_dir, "ampl", "modelo_bmtsp.mod")
    data_path = os.path.join(base_dir, "ampl", "datos_bmtsp.dat")
    
    # Comando AMPL
    cmd = f"""
    model \"{model_path}";
    data \"{data_path}";
    option solver cplex;
    solve;
    display Total_Distance;
    """
    
    print("Ejecutando AMPL con el comando:")
    print(cmd)
    
    try:
        # Usar la ruta completa a AMPL
        ampl_path = r"D:\DEV\AMPL\ampl.exe"
        print(f"\nIntentando ejecutar: {ampl_path}")
        
        result = subprocess.run(
            [ampl_path],
            input=cmd,
            text=True,
            capture_output=True
        )
        
        print("\n=== SALIDA ===")
        print(result.stdout)
        
        if result.stderr:
            print("\n=== ERRORES ===")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error al ejecutar AMPL: {e}")

if __name__ == "__main__":
    test_ampl()
