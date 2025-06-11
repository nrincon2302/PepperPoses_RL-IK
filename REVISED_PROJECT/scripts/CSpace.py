import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scripts.robot_graph import plot_robot, LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS, calculate_joint_positions


def generate_workspace_points(side='Left', n_samples=10):
    """
    Genera una nube de puntos que representa el espacio de trabajo del brazo.
    
    Args:
        side (str): 'Left' o 'Right' para indicar el brazo
        n_samples (int): Número de muestras por dimensión
        
    Returns:
        np.array: Array de puntos (x,y,z) alcanzables por el efector final
        list: Lista de diccionarios con los ángulos correspondientes
    """
    # Seleccionar límites según el lado
    limits = LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
    prefix = 'L' if side == 'Left' else 'R'
    
    # Generar valores para cada ángulo
    shoulder_pitch = np.linspace(limits[f'{prefix}ShoulderPitch'][0], limits[f'{prefix}ShoulderPitch'][1], n_samples)
    shoulder_roll = np.linspace(limits[f'{prefix}ShoulderRoll'][0], limits[f'{prefix}ShoulderRoll'][1], n_samples)
    elbow_yaw = np.linspace(limits[f'{prefix}ElbowYaw'][0], limits[f'{prefix}ElbowYaw'][1], n_samples)
    elbow_roll = np.linspace(limits[f'{prefix}ElbowRoll'][0], limits[f'{prefix}ElbowRoll'][1], n_samples)
    wrist_yaw = np.linspace(limits[f'{prefix}WristYaw'][0], limits[f'{prefix}WristYaw'][1], n_samples)
    
    points = []
    configurations = []
    
    # Muestrear el espacio de configuraciones
    for sp in shoulder_pitch:
        for sr in shoulder_roll:
            for ey in elbow_yaw:
                for er in elbow_roll:
                    for wy in wrist_yaw:
                        angles = {
                            f'{prefix}ShoulderPitch': sp,
                            f'{prefix}ShoulderRoll': sr,
                            f'{prefix}ElbowYaw': ey,
                            f'{prefix}ElbowRoll': er,
                            f'{prefix}WristYaw': wy
                        }
                        
                        # Calcular posición final
                        positions = calculate_joint_positions(side, angles)
                        end_effector = positions[-1]  # Última posición = efector final
                        points.append(end_effector)
                        configurations.append(angles)
    
    return np.array(points), configurations


def plot_workspace():
    """
    Visualiza el espacio de trabajo de ambos brazos superpuesto con el robot en pose estándar.
    """
    # Generar puntos del espacio de trabajo (usar menos muestras para visualización)
    left_points, _ = generate_workspace_points('Left', n_samples=5)
    right_points, _ = generate_workspace_points('Right', n_samples=5)
    
    # Imprimir estadísticas y puntos seleccionados
    print(f"\nPuntos totales generados:")
    print(f"Brazo izquierdo: {len(left_points)}")
    print(f"Brazo derecho: {len(right_points)}")
    
    # Seleccionar 20 puntos distribuidos uniformemente
    left_indices = np.linspace(0, len(left_points)-1, 20, dtype=int)
    right_indices = np.linspace(0, len(right_points)-1, 20, dtype=int)
    
    print("\n20 Puntos del espacio de trabajo del brazo izquierdo (x, y, z) en metros:")
    print("-" * 60)
    for i, idx in enumerate(left_indices):
        print(f"Punto {i+1:2d}: ({left_points[idx][0]:7.3f}, {left_points[idx][1]:7.3f}, {left_points[idx][2]:7.3f})")
    
    print("\n20 Puntos del espacio de trabajo del brazo derecho (x, y, z) en metros:")
    print("-" * 60)
    for i, idx in enumerate(right_indices):
        print(f"Punto {i+1:2d}: ({right_points[idx][0]:7.3f}, {right_points[idx][1]:7.3f}, {right_points[idx][2]:7.3f})")
    
    # Graficar robot en pose estándar con puntos superpuestos
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar el robot en pose estándar
    plot_robot(ax=ax)
    
    # Agregar nube de puntos con puntos más grandes
    ax.scatter(left_points[:, 0], left_points[:, 1], left_points[:, 2], 
              c='blue', alpha=0.2, s=4, label='Left Arm Workspace')
    ax.scatter(right_points[:, 0], right_points[:, 1], right_points[:, 2], 
              c='red', alpha=0.2, s=4, label='Right Arm Workspace')
    
    plt.show()


if __name__ == "__main__":
    plot_workspace()
