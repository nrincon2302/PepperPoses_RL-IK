# scripts/CSpace.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necesario para projection='3d'
import itertools

# Importar la cinemática y el entorno para los parámetros
from kinematics.forward import forward_kinematics
from envs.pepper_arm_env import PepperArmEnv5DOF # Para obtener parámetros y límites

# Instanciar el entorno para obtener sus parámetros
# No necesitamos renderizarlo ni nada, solo acceder a 'params' y 'joint_limits'
try:
    temp_env = PepperArmEnv5DOF()
    ROBOT_PARAMS = temp_env.params
    JOINT_LIMITS_LOW = temp_env.joint_limits_low
    JOINT_LIMITS_HIGH = temp_env.joint_limits_high
    NUM_JOINTS = temp_env.num_joints
    del temp_env
except Exception as e:
    print(f"Error instanciando PepperArmEnv5DOF para CSpace: {e}")
    print("Usando valores por defecto para CSpace (pueden no ser correctos).")
    # Valores de fallback si la importación/instanciación falla
    ROBOT_PARAMS = {
        'shoulder_offset_y': 0.14974,
        'upper_arm_length': 0.18120,
        'elbow_offset_y': 0.01500,
        'lower_arm_length': 0.15000
    }
    JOINT_LIMITS_LOW = np.array([-2.0857, -1.5620, -2.0857, 0.0087, -1.8239])
    JOINT_LIMITS_HIGH = np.array([2.0857, -0.0087, 2.0857, 1.5620, 1.8239])
    NUM_JOINTS = 5


def generate_workspace_points(num_samples_per_joint=5):
    """
    Genera puntos del espacio de trabajo muestreando configuraciones articulares.
    Args:
        num_samples_per_joint (int): Número de muestras a tomar para cada articulación
                                     dentro de sus límites. Cuidado, el número total de
                                     puntos es (num_samples_per_joint)^NUM_JOINTS.
    Returns:
        np.ndarray: Array de puntos (N, 3) en el espacio cartesiano.
    """
    if NUM_JOINTS > 3 and num_samples_per_joint > 5: # Prevenir explosión combinatoria
        print(f"Advertencia: {num_samples_per_joint}^{NUM_JOINTS} = {num_samples_per_joint**NUM_JOINTS} puntos a calcular. Esto puede tardar.")
        if NUM_JOINTS > 4 and num_samples_per_joint > 3:
             print("Reduciendo num_samples_per_joint a 3 para 5+ DOF para evitar bloqueo.")
             num_samples_per_joint = 3
        elif NUM_JOINTS > 3 and num_samples_per_joint > 4:
             print("Reduciendo num_samples_per_joint a 4 para 4 DOF.")
             num_samples_per_joint = 4


    # Crear un espacio lineal para cada articulación
    joint_samples = []
    for i in range(NUM_JOINTS):
        joint_samples.append(np.linspace(JOINT_LIMITS_LOW[i], JOINT_LIMITS_HIGH[i], num_samples_per_joint))

    workspace_points = []
    
    # Iterar sobre todas las combinaciones de ángulos articulares
    # itertools.product genera el producto cartesiano de los rangos de las articulaciones
    total_combinations = num_samples_per_joint**NUM_JOINTS
    print(f"Calculando {total_combinations} puntos del espacio de trabajo...")
    
    count = 0
    for joint_config in itertools.product(*joint_samples):
        q = np.array(joint_config, dtype=np.float32)
        # Calcular cinemática directa
        try:
            eff_pos, _ = forward_kinematics(q, ROBOT_PARAMS)
            workspace_points.append(eff_pos)
        except Exception as e:
            print(f"Error en cinemática directa para q={q}: {e}")
        count +=1
        if count % (total_combinations // 100 + 1) == 0 : # Imprimir progreso aprox cada 1%
             print(f"Progreso: {count*100/total_combinations:.2f}%", end='\r')
    print("\nCálculo completado.")
    return np.array(workspace_points)


def plot_workspace(points, title='Espacio de Trabajo del Brazo 5DOF'):
    if points.shape[0] == 0:
        print("No se generaron puntos para graficar.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Usar un subconjunto de puntos si hay demasiados para graficar eficientemente
    max_points_to_plot = 50000
    if points.shape[0] > max_points_to_plot:
        print(f"Mostrando {max_points_to_plot} de {points.shape[0]} puntos para mejorar rendimiento.")
        indices = np.random.choice(points.shape[0], max_points_to_plot, replace=False)
        points_to_plot = points[indices]
    else:
        points_to_plot = points

    ax.scatter(points_to_plot[:,0], points_to_plot[:,1], points_to_plot[:,2], s=2, c=points_to_plot[:,2], cmap='viridis', alpha=0.3)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Para mejorar la visualización, ajustar límites si es necesario
    ax.set_xlim([np.min(points[:,0]), np.max(points[:,0])])
    ax.set_ylim([np.min(points[:,1]), np.max(points[:,1])])
    ax.set_zlim([np.min(points[:,2]), np.max(points[:,2])])
    
    # Forzar aspecto igual para que las esferas parezcan esferas
    # Comentado porque puede hacer que la gráfica sea muy pequeña si los rangos son dispares
    # ax.set_box_aspect([ub - lb for lb, ub in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())])

    plt.show()

if __name__ == '__main__':
    # Para un brazo 5DOF, num_samples_per_joint=5 significa 5^5 = 3125 puntos. Manageable.
    # num_samples_per_joint=7 significa 7^5 = 16807 puntos. Aún manageable.
    # num_samples_per_joint=10 significa 10^5 = 100,000 puntos. Puede empezar a ser lento.
    
    # ¡CUIDADO! num_samples_per_joint=X para Y articulaciones da X^Y puntos.
    # Para 5DOF:
    # 3 -> 243
    # 4 -> 1024
    # 5 -> 3125
    # 7 -> 16807
    # 10 -> 100000
    print("Generando espacio de trabajo...")
    # Un valor más bajo para una prueba rápida
    workspace_pts = generate_workspace_points(num_samples_per_joint=1000) 
    if workspace_pts.size > 0:
        plot_workspace(workspace_pts)
    else:
        print("No se pudieron generar puntos del espacio de trabajo.")