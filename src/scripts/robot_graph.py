# =======================================
# Este script genera y visualiza el espacio de trabajo del brazo izquierdo o derecho de un robot Pepper.
# Se basa en la cinemática directa y los parámetros físicos del robot.
# Los puntos generados representan las posiciones alcanzables por el efector final del brazo.
# El brazo se dibuja en un espacio 3D, mostrando las trayectorias posibles.
# =======================================

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from kinematics.forward import get_arm_joints_positions

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


def create_cylinder(p1, p2, R, n=100):
    """
    Crea los puntos para dibujar un cilindro entre dos puntos.
    
    Args:
        p1 (np.array): Punto inicial del cilindro
        p2 (np.array): Punto final del cilindro
        R (float): Radio del cilindro
        n (int): Número de puntos para la discretización
        
    Returns:
        tuple: Arrays X, Y, Z con los puntos de la superficie del cilindro
    """
    v = p2 - p1
    mag = np.linalg.norm(v)
    if mag == 0:
        return None, None
    
    # Vector unitario en dirección del cilindro
    v = v / mag
    
    # Crear vector perpendicular arbitrario
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    
    # Vectores base para el círculo
    n1 = np.cross(v, not_v)
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    
    # Puntos del círculo
    t = np.linspace(0, 2*np.pi, n)
    x = R * np.cos(t)
    y = R * np.sin(t)
    
    # Crear superficie del cilindro
    XX = np.zeros((2, n))
    YY = np.zeros((2, n))
    ZZ = np.zeros((2, n))
    
    for i in range(n):
        point = p1 + x[i]*n1 + y[i]*n2
        XX[0, i] = point[0]
        YY[0, i] = point[1]
        ZZ[0, i] = point[2]
        point = p2 + x[i]*n1 + y[i]*n2
        XX[1, i] = point[0]
        YY[1, i] = point[1]
        ZZ[1, i] = point[2]
    
    return XX, YY, ZZ

# =======================================
# Funciones de Visualización
# =======================================
def draw_robot_body(ax):
    """
    Dibuja una representación sólida del cuerpo completo del robot.
    
    Args:
        ax (Axes3D): Objeto de ejes 3D de matplotlib
    """
    # Dimensiones del robot
    torso_width = 0.25
    torso_depth = 0.30
    n_points = 15
    
    # Alturas de las secciones del robot
    base_height = ROBOT_BODY_PARAMS['WheelRadius'] + ROBOT_BODY_PARAMS['ThighLength'] + ROBOT_BODY_PARAMS['TibiaLength']
    torso_height = 0.3
    total_height = base_height + ROBOT_BODY_PARAMS['HipOffsetZ'] + torso_height
    
    # Base (parte inferior trapezoidal)
    base_top_width = torso_width * 0.9  # La base superior será 90% del ancho del torso
    base_top_depth = torso_depth * 0.9
    base_bottom_width = base_top_width * 1.3
    base_bottom_depth = base_top_depth * 1.3
    
    # Crear puntos para todo el cuerpo (base + torso como una sola superficie)
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(0, total_height, n_points)
    theta, z = np.meshgrid(theta, z)
    
    # Generar superficie continua
    factor = np.zeros_like(z)
    for i in range(len(z)):
        height = z[i,0]
        if height < base_height:
            # Parte trapezoidal (base)
            factor[i] = (base_height - height) / base_height
        else:
            # Parte cilíndrica (torso)
            factor[i] = 0
            
    r_x = (base_bottom_width/2 * factor + base_top_width/2 * (1-factor))
    r_y = (base_bottom_depth/2 * factor + base_top_depth/2 * (1-factor))
    x = r_x * np.cos(theta)
    y = r_y * np.sin(theta)
    
    # Dibujar cuerpo completo
    ax.plot_surface(x, y, z, alpha=0.1, color='lightgray', shade=True)
    
    # Cabeza (esfera)
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    r = 0.1
    head_center = np.array([0, 0, total_height + ROBOT_BODY_PARAMS['NeckOffsetZ']/2])
    
    x_head = head_center[0] + r * np.outer(np.cos(u), np.sin(v))
    y_head = head_center[1] + r * np.outer(np.sin(u), np.sin(v))
    z_head = head_center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_head, y_head, z_head, alpha=0.1, color='lightgray', shade=True)


def plot_robot(left_angles=None, right_angles=None, ax=None):
    """
    Visualiza el robot con los ángulos especificados.
    
    Args:
        left_angles (dict): Ángulos del brazo izquierdo en radianes
        right_angles (dict): Ángulos del brazo derecho en radianes
        ax (Axes3D): Eje 3D existente donde dibujar. Si es None, crea uno nuevo.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True
    else:
        show_plot = False
    
    # Calcula posiciones para ambos brazos
    left_positions = calculate_joint_positions('Left', left_angles)
    right_positions = calculate_joint_positions('Right', right_angles)
    
    # Dibuja el cuerpo del robot
    draw_robot_body(ax)
    
    # Radio de los cilindros para los brazos
    R = 0.02
    
    # Dibuja los brazos con volumen
    for i in range(2, len(left_positions)-1):
        # Brazo izquierdo
        X, Y, Z = create_cylinder(left_positions[i], left_positions[i+1], R)
        if X is not None:
            ax.plot_surface(X, Y, Z, alpha=0.5, color='blue')
        
        # Brazo derecho
        X, Y, Z = create_cylinder(right_positions[i], right_positions[i+1], R)
        if X is not None:
            ax.plot_surface(X, Y, Z, alpha=0.5, color='red')
    
    # Dibuja las líneas centrales
    ax.plot(left_positions[2:, 0], left_positions[2:, 1], left_positions[2:, 2], 'b-', linewidth=1, label='Left Arm')
    ax.plot(right_positions[2:, 0], right_positions[2:, 1], right_positions[2:, 2], 'r-', linewidth=1, label='Right Arm')
    
    # Marca los joints
    ax.scatter(left_positions[2:, 0], left_positions[2:, 1], left_positions[2:, 2], c='blue', marker='o')
    ax.scatter(right_positions[2:, 0], right_positions[2:, 1], right_positions[2:, 2], c='red', marker='o')
    
    # Marca el origen
    ax.scatter([0], [0], [0], c='black', marker='x', s=100, label='Origin')
    
    # Configuración del plot
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Pepper Robot Arms - Standard Position')
    ax.legend()
    
    # Ajusta los límites
    max_range_xy = 0.4  # Ajustado para mejor visualización
    ax.set_xlim([-max_range_xy, max_range_xy])
    ax.set_ylim([-max_range_xy, max_range_xy])
    ax.set_zlim([0, 1.2])
    
    # Ajusta la vista
    ax.set_aspect('equal')
    ax.view_init(elev=20, azim=45)
    
    if show_plot:
        plt.show()
    
    return ax



if __name__ == "__main__":
    # Ejemplo 1: Posición por defecto (todos los ángulos en 0)
    plot_robot()
    
    # Ejemplo 2: Brazos levantados
    left_angles = {
        'LShoulderPitch': -np.pi/4,  # -45 grados
        'LShoulderRoll': np.pi/6,    # 30 grados
        'LElbowYaw': 0,
        'LElbowRoll': -np.pi/3,      # -60 grados
        'LWristYaw': 0
    }
    
    right_angles = {
        'RShoulderPitch': -np.pi/4,  # -45 grados
        'RShoulderRoll': -np.pi/6,   # -30 grados
        'RElbowYaw': 0,
        'RElbowRoll': np.pi/3,       # 60 grados
        'RWristYaw': 0
    }
    
    plot_robot(left_angles, right_angles)
