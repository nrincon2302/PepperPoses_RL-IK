# =======================================
# Cinemática directa del robot Pepper.
# Implementa las transformaciones necesarias para calcular las posiciones
# de los joints dado un conjunto de ángulos.
# =======================================

import numpy as np

def rotation_matrix_x(theta):
    """Matriz de rotación alrededor del eje X."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    """Matriz de rotación alrededor del eje Y."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    """Matriz de rotación alrededor del eje Z."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def transform_matrix(rotation, translation):
    """
    Crea matriz de transformación homogénea 4x4.
    
    Args:
        rotation (np.array): Matriz de rotación 3x3
        translation (np.array): Vector de traslación 3x1
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def get_arm_joints_positions(joint_angles, side='Left', origin_torso=np.zeros(3), link_vectors=None):
    """
    Calcula las posiciones de los joints para un conjunto de ángulos dado.
    
    Args:
        joint_angles (dict): Ángulos de los joints en radianes
        side (str): 'Left' o 'Right' para indicar el brazo
        origin_torso (np.array): Posición del origen del torso
        link_vectors (dict): Vectores de desplazamiento entre joints
    
    Returns:
        np.array: Array con las posiciones 3D de cada joint
    """
    # Inicialización
    positions = [np.array([0.0, 0.0, 0.0]), origin_torso]  # [origen_global, origen_torso]
    current_transform = np.eye(4)
    current_transform[:3, 3] = origin_torso
    
    # Prefijos según el lado
    prefix = 'L' if side == 'Left' else 'R'
    
    # Secuencia de transformaciones
    # ShoulderPitch (rotación en Y)
    T_shoulder_pitch = transform_matrix(
        rotation_matrix_y(joint_angles[f'{prefix}ShoulderPitch']),
        link_vectors[f'Torso_{prefix}ShoulderPitch']
    )
    current_transform = current_transform @ T_shoulder_pitch
    positions.append(current_transform[:3, 3])
    
    # ShoulderRoll (rotación en Z)
    T_shoulder_roll = transform_matrix(
        rotation_matrix_z(joint_angles[f'{prefix}ShoulderRoll']),
        link_vectors[f'{prefix}ShoulderPitch_{prefix}ShoulderRoll']
    )
    current_transform = current_transform @ T_shoulder_roll
    
    # ElbowYaw (rotación en X)
    T_elbow_yaw = transform_matrix(
        rotation_matrix_x(joint_angles[f'{prefix}ElbowYaw']),
        link_vectors[f'{prefix}ShoulderRoll_{prefix}ElbowYaw']
    )
    current_transform = current_transform @ T_elbow_yaw
    positions.append(current_transform[:3, 3])
    
    # ElbowRoll (rotación en Z)
    T_elbow_roll = transform_matrix(
        rotation_matrix_z(joint_angles[f'{prefix}ElbowRoll']),
        link_vectors[f'{prefix}ElbowYaw_{prefix}ElbowRoll']
    )
    current_transform = current_transform @ T_elbow_roll
    
    # WristYaw (rotación en X)
    T_wrist_yaw = transform_matrix(
        rotation_matrix_x(joint_angles[f'{prefix}WristYaw']),
        link_vectors[f'{prefix}ElbowRoll_{prefix}WristYaw']
    )
    current_transform = current_transform @ T_wrist_yaw
    positions.append(current_transform[:3, 3])
    
    # Hand
    T_hand = transform_matrix(
        np.eye(3),
        link_vectors[f'{prefix}WristYaw_{prefix}Hand']
    )
    current_transform = current_transform @ T_hand
    positions.append(current_transform[:3, 3])
    
    return np.array(positions)
