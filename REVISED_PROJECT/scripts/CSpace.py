import numpy as np
import matplotlib.pyplot as plt
from scripts.robot_graph import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS, calculate_joint_positions
from qibullet import SimulationManager
import pybullet as p

def generate_workspace_points(side='Left', n_samples=10):
    # (código sin cambios)
    limits = LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
    prefix = 'L' if side == 'Left' else 'R'
    shoulder_pitch = np.linspace(limits[f'{prefix}ShoulderPitch'][0], limits[f'{prefix}ShoulderPitch'][1], n_samples)
    shoulder_roll = np.linspace(limits[f'{prefix}ShoulderRoll'][0], limits[f'{prefix}ShoulderRoll'][1], n_samples)
    elbow_yaw = np.linspace(limits[f'{prefix}ElbowYaw'][0], limits[f'{prefix}ElbowYaw'][1], n_samples)
    elbow_roll = np.linspace(limits[f'{prefix}ElbowRoll'][0], limits[f'{prefix}ElbowRoll'][1], n_samples)
    wrist_yaw = np.linspace(limits[f'{prefix}WristYaw'][0], limits[f'{prefix}WristYaw'][1], n_samples)
    points, configurations = [], []
    for sp in shoulder_pitch:
        for sr in shoulder_roll:
            for ey in elbow_yaw:
                for er in elbow_roll:
                    for wy in wrist_yaw:
                        angles = {
                            f'{prefix}ShoulderPitch': sp, f'{prefix}ShoulderRoll': sr,
                            f'{prefix}ElbowYaw': ey, f'{prefix}ElbowRoll': er, f'{prefix}WristYaw': wy
                        }
                        positions = calculate_joint_positions(side, angles)
                        end_effector = positions[-1]
                        points.append(end_effector)
                        configurations.append(angles)
    return np.array(points), configurations


def plot_workspace_qibullet():
    """Visualiza el espacio de trabajo en una simulación de qiBullet."""
    print("Generando puntos del workspace para visualización en qiBullet (puede tardar)...")
    left_points, _ = generate_workspace_points('Left', n_samples=4)
    right_points, _ = generate_workspace_points('Right', n_samples=4)

    # Iniciar simulación
    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=True)
    pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)

    print("Dibujando workspace en la simulación...")
    # --- CAMBIO AQUÍ: Usamos el módulo pybullet (p) directamente ---
    for point in left_points:
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[0, 0, 1, 0.3],
            physicsClientId=client # Buena práctica especificar el cliente
        )
        p.createMultiBody(
            baseMass=0,  # Estático
            baseVisualShapeIndex=visual_shape_id,
            basePosition=point.tolist(),
            physicsClientId=client
        )
    
    for point in right_points:
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[1, 0, 0, 0.3],
            physicsClientId=client
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=point.tolist(),
            physicsClientId=client
        )
    # --- FIN DEL CAMBIO ---

    print("\nWorkspace visualizado en la ventana de qiBullet.")
    print("Cierra la ventana de simulación o presiona Ctrl+C en la terminal para salir.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        simulation_manager.stopSimulation(client)

if __name__ == "__main__":
    plot_workspace_qibullet()