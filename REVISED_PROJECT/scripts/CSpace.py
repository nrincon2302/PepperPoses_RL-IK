import numpy as np
import matplotlib.pyplot as plt
from qibullet import SimulationManager
import pybullet as p
from kinematics.forward import get_arm_joints_positions
import os # <--- AÑADIDO

# =======================================
# Parámetros Físicos del Robot
# =======================================

# Parámetros de longitud de los brazos del robot
LEFT_ARM_LINKS = {
    'UpperArmLength': 0.18120,
    'LowerArmLength': 0.15000,
    'ShoulderOffsetY': 0.14974,
    'ElbowOffsetY': 0.01500,
    'HandOffsetX': 0.06950,
    'HandOffsetZ': 0.03030
}

RIGHT_ARM_LINKS = {
    'UpperArmLength': 0.18120,
    'LowerArmLength': 0.15000,
    'ShoulderOffsetY': 0.14974,
    'ElbowOffsetY': 0.01500,
    'HandOffsetX': 0.06950,
    'HandOffsetZ': 0.03030
}

# Parámetros del cuerpo del robot - Dimensiones físicas
ROBOT_BODY_PARAMS = {
    'WheelRadius': 0.07000,
    'TibiaLength': 0.26400,
    'ThighLength': 0.26800,
    'HipOffsetZ': 0.07900,
    'WaistOffsetZ': 0.13900,
    'NeckOffsetZ': 0.16990
}


# =======================================
# Vectores de Desplazamiento y Límites
# =======================================

# Desplazamientos por cada link respecto al origen anterior (x,y,z)
LEFT_LINK_DISPLACEMENT_VECTOR = {
    'Torso_LShoulderPitch': np.array([-0.0570, 0.14974, 0.08682]),
    'LShoulderPitch_LShoulderRoll': np.array([0.0, 0.0, 0.0]),
    'LShoulderRoll_LElbowYaw': np.array([0.18120, 0.01500, 0.00013]),
    'LElbowYaw_LElbowRoll': np.array([0.0, 0.0, 0.0]),
    'LElbowRoll_LWristYaw': np.array([0.15000, 0.02360, 0.02284]),
    'LWristYaw_LHand': np.array([0.06950, 0.0, -0.03030])
}

RIGHT_LINK_DISPLACEMENT_VECTOR = {
    'Torso_RShoulderPitch': np.array([-0.0570, -0.14974, 0.08682]),
    'RShoulderPitch_RShoulderRoll': np.array([0.0, 0.0, 0.0]),
    'RShoulderRoll_RElbowYaw': np.array([0.18120, -0.01500, 0.00013]),
    'RElbowYaw_RElbowRoll': np.array([0.0, 0.0, 0.0]),
    'RElbowRoll_RWristYaw': np.array([0.15000, 0.02360, 0.02284]),
    'RWristYaw_RHand': np.array([0.06950, 0.0, -0.03030])
}


# Valores límite de los Joints (en grados)
LEFT_JOINT_LIMITS = {
    'LShoulderPitch': (-2.0857, 2.0857),
    'LShoulderRoll': (0.0087, 1.5620),
    'LElbowYaw': (-2.0857, 2.0857),
    'LElbowRoll': (-1.3614, -0.0087), # -78° para proteger límite condicional
    'LWristYaw': (-1.8239, 1.8239)
}

RIGHT_JOINT_LIMITS = {
    'RShoulderPitch': (-2.0857, 2.0857),
    'RShoulderRoll': (-1.5620, -0.0087),
    'RElbowYaw': (-2.0857, 2.0857),
    'RElbowRoll': (0.0087, 1.3614), # 78° para proteger límite condicional
    'RWristYaw': (-1.8239, 1.8239)
}


ORIGIN = np.array([0.0, 0.0, 0.0])  # Origen del sistema de coordenadas global
ORIGIN_TORSO = np.array([
    0.00,
    0.00,
    ROBOT_BODY_PARAMS['WaistOffsetZ'] + ROBOT_BODY_PARAMS['HipOffsetZ'] + ROBOT_BODY_PARAMS['TibiaLength'] + 
    ROBOT_BODY_PARAMS['ThighLength'] + ROBOT_BODY_PARAMS['WheelRadius']
]) # Posición del origen del torso en el sistema de coordenadas global


# =======================================
# Funciones de Cálculo
# =======================================
def calculate_joint_positions(side='Left', joint_angles=None):
    """
    Calcula las posiciones de los joints en el espacio 3D.
    
    Args:
        side (str): 'Left' o 'Right' para indicar el brazo a calcular
        joint_angles (dict): Ángulos de los joints en radianes. Si es None, usa 0s.
        
    Returns:
        np.array: Array con las posiciones 3D de cada joint
    """
    if joint_angles is None:
        # Ángulos por defecto (todos en 0)
        prefix = 'L' if side == 'Left' else 'R'
        joint_angles = {
            f'{prefix}ShoulderPitch': 0.0,
            f'{prefix}ShoulderRoll': 0.0,
            f'{prefix}ElbowYaw': 0.0,
            f'{prefix}ElbowRoll': 0.0,
            f'{prefix}WristYaw': 0.0
        }
    
    displacement_vectors = LEFT_LINK_DISPLACEMENT_VECTOR if side == 'Left' else RIGHT_LINK_DISPLACEMENT_VECTOR
    return get_arm_joints_positions(joint_angles, side, ORIGIN_TORSO, displacement_vectors)


# =======================================
# Funciones de Generación del Workspace
# =======================================
def generate_workspace_points(side='Left', n_samples=10):
    """
    Genera puntos del espacio de trabajo para el brazo especificado.
    Utiliza un sistema de caché para evitar la regeneración en ejecuciones sucesivas.
    Los puntos se generan en una cuadrícula uniforme dentro de los límites de las articulaciones del brazo.
    """
    cache_dir = "workspace_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"workspace_{side.lower()}_{n_samples}.npy")

    # Si el archivo de caché existe, cargarlo directamente
    if os.path.exists(cache_file):
        print(f"Cargando workspace desde el caché: {cache_file}")
        points = np.load(cache_file)
        # Las configuraciones no se guardan en caché para ahorrar espacio. Se devuelve una lista vacía.
        return points, []

    print(f"Generando puntos del workspace para '{side}' con n_samples={n_samples}. Esto puede tardar...")
    limits = LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
    prefix = 'L' if side == 'Left' else 'R'
    # Consultar los límites de las articulaciones
    shoulder_pitch = np.linspace(limits[f'{prefix}ShoulderPitch'][0], limits[f'{prefix}ShoulderPitch'][1], n_samples)
    shoulder_roll = np.linspace(limits[f'{prefix}ShoulderRoll'][0], limits[f'{prefix}ShoulderRoll'][1], n_samples)
    elbow_yaw = np.linspace(limits[f'{prefix}ElbowYaw'][0], limits[f'{prefix}ElbowYaw'][1], n_samples)
    elbow_roll = np.linspace(limits[f'{prefix}ElbowRoll'][0], limits[f'{prefix}ElbowRoll'][1], n_samples)
    wrist_yaw = np.linspace(limits[f'{prefix}WristYaw'][0], limits[f'{prefix}WristYaw'][1], n_samples)
    # Generar combinaciones de ángulos
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

    points_np = np.array(points)
    # Guardar los puntos generados en el caché
    print(f"Guardando {len(points_np)} puntos del workspace en: {cache_file}")
    np.save(cache_file, points_np)
    
    return points_np, configurations


def plot_workspace_qibullet():
    """
    Visualiza el espacio de trabajo en una simulación de qiBullet.
    Genera puntos del workspace para ambos brazos y los dibuja en la simulación.
    """
    print("Generando puntos del workspace para visualización en qiBullet (usará caché si está disponible)...")
    left_points, _ = generate_workspace_points('Left', n_samples=4)
    right_points, _ = generate_workspace_points('Right', n_samples=4)

    # Iniciar simulación
    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=True)
    pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)

    print("Dibujando workspace en la simulación...")
    for point in left_points:
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[0, 0, 1, 0.3],
            physicsClientId=client
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