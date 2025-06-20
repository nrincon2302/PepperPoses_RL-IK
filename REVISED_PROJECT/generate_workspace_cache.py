import os
import numpy as np
from scripts.CSpace import generate_workspace_points
import time

CACHE_DIR = "workspace_cache"
SAMPLES_PER_DIM = 8

def create_cache():
    """
    Genera los puntos del workspace para ambos brazos y los guarda en archivos .npy
    Este script debe ejecutarse una sola vez antes de cualquier entrenamiento.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Directorio de caché creado en: {CACHE_DIR}")

    for side in ['Left', 'Right']:
        cache_file = os.path.join(CACHE_DIR, f"workspace_{side.lower()}_{SAMPLES_PER_DIM}.npy")
        
        if os.path.exists(cache_file):
            print(f"El archivo de caché para el brazo '{side}' ya existe. Saltando.")
            continue

        print(f"Generando workspace para el brazo '{side}' (n_samples={SAMPLES_PER_DIM})...")
        print("Esto puede tardar varios minutos, pero solo se hace una vez.")
        
        start_time = time.time()
        # Aquí es donde ocurre el cálculo costoso
        points, _ = generate_workspace_points(side=side, n_samples=SAMPLES_PER_DIM)
        end_time = time.time()
        
        np.save(cache_file, points)
        print(f"Workspace para '{side}' guardado en '{cache_file}'.")
        print(f"Tiempo de generación: {end_time - start_time:.2f} segundos.")

if __name__ == "__main__":
    create_cache()
    print("\nGeneración de caché completada. Ya puedes ejecutar el entrenamiento.")