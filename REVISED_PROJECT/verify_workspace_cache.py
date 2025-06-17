import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Opcional: para visualizar con el robot
try:
    import pybullet as p
    from qibullet import SimulationManager
    QIBULLET_AVAILABLE = True
except ImportError:
    QIBULLET_AVAILABLE = False
    print("Advertencia: qiBullet no está instalado. La visualización con el robot no estará disponible.")

CACHE_DIR = "workspace_cache"

def verify_and_visualize(side='Left', n_samples=8, use_qibullet=False):
    """
    Carga, analiza y visualiza el archivo de caché del workspace.
    """
    cache_file = os.path.join(CACHE_DIR, f"workspace_{side.lower()}_{n_samples}.npy")

    # --- 1. Verificación de Existencia ---
    if not os.path.exists(cache_file):
        print(f"Error: El archivo de caché no fue encontrado en '{cache_file}'")
        print("Por favor, ejecuta 'generate_workspace_cache.py' primero.")
        return

    print(f"--- Verificando caché: {cache_file} ---")

    # --- 2. Carga y Análisis Estadístico ---
    try:
        points = np.load(cache_file)
        print(f"Cargado exitosamente. Forma del array: {points.shape}")
        
        if len(points.shape) != 2 or points.shape[1] != 3:
            print(f"Error: El array no tiene la forma esperada (N, 3). La forma actual es {points.shape}.")
            return
        
        num_points = points.shape[0]
        expected_points = n_samples ** 5
        print(f"Número de puntos en el caché: {num_points} (Esperados: {expected_points})")
        if num_points != expected_points:
            print("Advertencia: El número de puntos no coincide con el esperado. "
                  "Asegúrate de que 'SAMPLES_PER_DIM' en el generador y el verificador sea el mismo.")

        # Imprimir rangos de coordenadas
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        print("\nRangos de Coordenadas (min, max):")
        print(f"  X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]")
        print(f"  Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
        print(f"  Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
        print("-" * 35)

    except Exception as e:
        print(f"Error al cargar o analizar el archivo de caché: {e}")
        return

    # --- 3. Visualización 3D con Matplotlib ---
    print("\nGenerando visualización 3D con Matplotlib...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar la nube de puntos
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.1, s=5)
    
    ax.set_title(f"Visualización del Workspace Cache - Brazo {side}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_aspect('equal')
    # Ajustar vista para que se vea bien
    ax.view_init(elev=20, azim=45)
    plt.show()

    # --- 4. Visualización Opcional con qiBullet ---
    if use_qibullet and QIBULLET_AVAILABLE:
        print("\nGenerando visualización 3D con qiBullet (puede tardar en cargar)...")
        simulation_manager = SimulationManager()
        client = simulation_manager.launchSimulation(gui=True)
        pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)

        print("Dibujando puntos en la simulación...")
        for point in points[::50]:  # Dibujar solo una fracción para no sobrecargar
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 1, 0.5]
                ),
                basePosition=point.tolist()
            )
        
        print("\nVisualización en qiBullet lista. Presiona Ctrl+C en la terminal para salir.")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            pass
        finally:
            simulation_manager.stopSimulation(client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica y visualiza el caché del workspace.")
    parser.add_argument("--side", choices=['Left', 'Right'], default='Left', help="Qué brazo verificar.")
    parser.add_argument("--samples", type=int, default=8, help="Resolución de muestreo usada para generar el caché.")
    parser.add_argument("--qibullet", action='store_true', help="Activar visualización adicional con qiBullet.")
    args = parser.parse_args()

    verify_and_visualize(side=args.side, n_samples=args.samples, use_qibullet=args.qibullet)