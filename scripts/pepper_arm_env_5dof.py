import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

class PepperArmEnv5DOF(gym.Env):
    """
    Entorno Gymnasium para el brazo derecho 5DOF de Pepper (Cinemática Inversa).

    Objetivo: Alcanzar una posición (x, y, z) objetivo con el efector final (muñeca).

    DOF: RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw

    Espacio de Observación (13 dimensiones):
        - cos(j1), sin(j1)
        - cos(j2), sin(j2)
        - cos(j3), sin(j3)
        - cos(j4), sin(j4)
        - cos(j5), sin(j5)
        - target_x - current_x
        - target_y - current_y
        - target_z - current_z

    Espacio de Acción (5 dimensiones):
        - delta_j1, delta_j2, delta_j3, delta_j4, delta_j5 (cambios angulares)

    Recompensa:
        - Basada en la distancia al objetivo.
        - Bonus al estar cerca.
        - Penalización severa por violar límites articulares.

    Curriculum:
        - Rango inicial de ángulos y distancia del objetivo aumentan gradualmente.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_steps=250):
        super().__init__()

        # --- Dimensiones Físicas (convertidas a metros) ---
        self.ShoulderOffsetY = 0.14974 # Distancia del centro del cuerpo al hombro en Y
        self.UpperArmLength = 0.18120 # L1: Hombro (Pitch/Roll) a Codo (Yaw/Roll)
        self.ElbowOffsetY = 0.01500   # Offset del codo en Y local
        self.LowerArmLength = 0.15000 # L2: Codo a Muñeca (Yaw)
        self.HandOffsetX = 0.06950    # Offset X de la mano desde la muñeca (no usado si target es muñeca)
        self.HandOffsetZ = 0.03030    # Offset Z de la mano desde la muñeca (no usado si target es muñeca)
        # Punto objetivo = origen del frame de RWristYaw

        self.link_lengths = [self.UpperArmLength, self.LowerArmLength]
        self.max_reach = self.UpperArmLength + self.LowerArmLength # Aproximado, sin offsets

        # --- Límites Articulares (radianes) ---
        # RShoulderPitch (Y rotation)
        self.joint_limits_low = [-2.0857]
        self.joint_limits_high = [2.0857]
        # RShoulderRoll (Z rotation)
        self.joint_limits_low.append(-1.5620)
        self.joint_limits_high.append(-0.0087) # Límite superior muy restrictivo!
        # RElbowYaw (Z' rotation)
        self.joint_limits_low.append(-2.0857)
        self.joint_limits_high.append(2.0857)
        # RElbowRoll (X' rotation) - Usamos el rango más amplio posible [0.5, 89.5] deg
        # La dependencia con RElbowYaw se ignora por simplicidad, confiando en la penalización.
        self.joint_limits_low.append(0.0087) # 0.5 degrees
        self.joint_limits_high.append(1.5620) # 89.5 degrees
        # RWristYaw (X'' rotation)
        self.joint_limits_low.append(-1.8239)
        self.joint_limits_high.append(1.8239)

        self.joint_limits_low = np.array(self.joint_limits_low, dtype=np.float32)
        self.joint_limits_high = np.array(self.joint_limits_high, dtype=np.float32)
        self.num_joints = len(self.joint_limits_low)

        # --- Espacios de Acción y Observación ---
        action_max = 0.1 # Máximo cambio angular por paso (rad)
        self.action_space = spaces.Box(low=-action_max, high=action_max, shape=(self.num_joints,), dtype=np.float32)

        # Observación: [cos(j1..j5), sin(j1..j5), dx, dy, dz]
        obs_low = np.concatenate([np.full(self.num_joints * 2, -1.0), np.full(3, -self.max_reach * 2)])
        obs_high = np.concatenate([np.full(self.num_joints * 2, 1.0), np.full(3, self.max_reach * 2)])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(self.num_joints * 2 + 3,), dtype=np.float32)

        # --- Estado Interno ---
        self.joint_angles = np.zeros(self.num_joints, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.end_effector_pos = np.zeros(3, dtype=np.float32)

        # --- Parámetros del Episodio ---
        self.max_episode_steps = max_steps
        self.current_step = 0
        self.goal_threshold = 0.03 # Umbral de distancia para éxito (metros)
        self.reward_bonus_threshold = 0.1 # Umbral para bonus
        self.joint_limit_penalty = 10.0 # Penalización severa

        # --- Curriculum Learning ---
        self.curriculum_step = 0
        self.curriculum_max_steps = 10 # Número de etapas del curriculum
        self.initial_angle_range_factor = 0.1 # Factor inicial para rango de ángulos
        self.initial_target_range_factor = 0.2 # Factor inicial para rango de objetivo

        # --- Rendering ---
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.arm_line = None
        self.target_marker = None
        self.ef_marker = None

        # Añadir memoria de estados previos
        self.memory_size = 3  # Guardar los últimos 3 estados
        self.joints_memory = []
        self.positions_memory = []
        
        # Observación expandida: actuales + históricos + velocidades estimadas
        obs_dim = self.num_joints * 2 + 3 + (self.num_joints + 3) * (self.memory_size - 1)
        obs_low = np.full(obs_dim, -np.inf)
        obs_high = np.full(obs_dim, np.inf)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        

    def _get_transform(self, angle, axis, translate=None):
        """Crea una matriz de transformación homogénea 4x4."""
        T = np.eye(4)
        if axis == 'x':
            T[:3, :3] = R.from_euler('x', angle).as_matrix()
        elif axis == 'y':
            T[:3, :3] = R.from_euler('y', angle).as_matrix()
        elif axis == 'z':
            T[:3, :3] = R.from_euler('z', angle).as_matrix()
        else:
            raise ValueError(f"Eje de rotación desconocido: {axis}")
        if translate is not None:
            T[:3, 3] = translate
        return T

    def _forward_kinematics(self, angles):
        """Calcula la Cinemática Directa usando transformaciones homogéneas."""
        j1, j2, j3, j4, j5 = angles

        # Base frame en el origen del cuerpo, X adelante, Y izquierda, Z arriba
        # Transformación al origen de RShoulderPitch (solo offset Y negativo)
        T_base_shoulder = self._get_transform(0, 'x', [0, -self.ShoulderOffsetY, 0])

        # 1. RShoulderPitch (Rotación Y)
        T_sp = self._get_transform(j1, 'y')
        # 2. RShoulderRoll (Rotación Z)
        T_sr = self._get_transform(j2, 'z')
        # 3. Link UpperArm (Traslación X)
        T_l1 = self._get_transform(0, 'x', [self.UpperArmLength, 0, 0])
        # 4. Offset Codo (Traslación Y)
        T_eo = self._get_transform(0, 'x', [0, self.ElbowOffsetY, 0])
        # 5. RElbowYaw (Rotación Z')
        T_ey = self._get_transform(j3, 'z')
        # 6. RElbowRoll (Rotación X')
        T_er = self._get_transform(j4, 'x')
        # 7. Link LowerArm (Traslación X)
        T_l2 = self._get_transform(0, 'x', [self.LowerArmLength, 0, 0])
        # 8. RWristYaw (Rotación X'')
        T_wy = self._get_transform(j5, 'x')

        # Componer transformaciones
        T_final = T_base_shoulder @ T_sp @ T_sr @ T_l1 @ T_eo @ T_ey @ T_er @ T_l2 @ T_wy

        # La posición del efector final (origen del frame de RWristYaw) es la columna de traslación
        ee_pos = T_final[:3, 3]
        return ee_pos, T_final # Devolver también la matriz completa para render

    def _get_obs(self):
        """Calcula la observación actual con memoria."""
        cos_sin_angles = np.array([[math.cos(a), math.sin(a)] for a in self.joint_angles]).flatten()
        delta_pos = self.target_pos - self.end_effector_pos
        
        # Añadir la posición y ángulos actuales a la memoria
        self.joints_memory.append(self.joint_angles.copy())
        self.positions_memory.append(self.end_effector_pos.copy())
        
        # Mantener solo los últimos memory_size estados
        if len(self.joints_memory) > self.memory_size:
            self.joints_memory.pop(0)
        if len(self.positions_memory) > self.memory_size:
            self.positions_memory.pop(0)
        
        # Inicializar la observación con los valores actuales
        obs = np.concatenate([cos_sin_angles, delta_pos])
        
        # Añadir información histórica
        for i in range(1, self.memory_size):
            if i < len(self.joints_memory):
                # Calcular diferencias para estimar velocidades
                angle_diff = self.joints_memory[-1] - self.joints_memory[-i-1]
                pos_diff = self.positions_memory[-1] - self.positions_memory[-i-1]
                obs = np.concatenate([obs, angle_diff, pos_diff])
            else:
                # Rellenar con ceros si no hay suficiente historia
                padding = np.zeros(self.num_joints + 3)
                obs = np.concatenate([obs, padding])
        
        return obs.astype(np.float32)

    def _get_info(self):
        """Devuelve información adicional."""
        distance = np.linalg.norm(self.target_pos - self.end_effector_pos)
        return {
            "distance": distance,
            "target_pos": self.target_pos.copy(),
            "current_pos": self.end_effector_pos.copy(),
            "current_angles": self.joint_angles.copy(),
            "joint_limits_low": self.joint_limits_low,
            "joint_limits_high": self.joint_limits_high,
            "curriculum_step": self.curriculum_step
        }


    def _check_joint_limits(self, angles):
        """Verifica si algún ángulo está fuera de los límites."""
        return np.any(angles < self.joint_limits_low) or np.any(angles > self.joint_limits_high)


    def calculate_reward(self, new_distance, prev_distance, action):
        # Recompensa por mejora de la distancia
        distance_improvement = prev_distance - new_distance
        improvement_reward = distance_improvement * 20.0  # Premiar la mejora
        
        # Recompensa por proximidad (función suave)
        proximity_factor = 1.0 / (1.0 + 10.0 * new_distance**2)
        proximity_reward = 10.0 * proximity_factor
        
        # Penalización por movimientos excesivos
        energy_penalty = -0.05 * np.sum(np.square(action))
        
        # Recompensa total
        reward = improvement_reward + proximity_reward + energy_penalty
        
        return reward


    def step(self, action):
        # Aplicar acción (cambio de ángulos)
        self.joint_angles = self.joint_angles + np.array(action, dtype=np.float32)

        # Verificar límites articulares
        violated_limits = self._check_joint_limits(self.joint_angles)
        limit_penalty = 0.0
        if violated_limits:
            limit_penalty = self.joint_limit_penalty
            # Clip angles back into limits to prevent instability
            self.joint_angles = np.clip(self.joint_angles, self.joint_limits_low, self.joint_limits_high)

        # Calcular nueva posición del efector final
        self.end_effector_pos, _ = self._forward_kinematics(self.joint_angles)

        # Calcular distancia al objetivo
        distance_to_target = np.linalg.norm(self.target_pos - self.end_effector_pos)

        # Calcular recompensa
        self.prev_distance = distance_to_target
        reward = self.calculate_reward(distance_to_target, self.prev_distance, action) - limit_penalty

        # Determinar si el episodio terminó (terminated)
        terminated = distance_to_target <= self.goal_threshold
        if terminated:
             reward += 50.0 # Bonus grande por alcanzar el objetivo

        # Determinar si el episodio fue truncado (truncated)
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        # Obtener observación e info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info


    def generate_target(self):
        # Definir regiones del espacio de trabajo
        regions = [
            {"name": "frontal_cerca", "weight": 0.3, "rho": [0.1, 0.2], "theta": [0.7, 1.0], "phi": [-0.5, 0.5]},
            {"name": "lateral_derecha", "weight": 0.2, "rho": [0.1, 0.25], "theta": [0.7, 1.2], "phi": [0.5, 1.5]},
            {"name": "lateral_izquierda", "weight": 0.2, "rho": [0.1, 0.25], "theta": [0.7, 1.2], "phi": [-1.5, -0.5]},
            {"name": "superior", "weight": 0.15, "rho": [0.15, 0.25], "theta": [0.3, 0.7], "phi": [-0.7, 0.7]},
            {"name": "lejano", "weight": 0.15, "rho": [0.25, 0.35], "theta": [0.7, 1.0], "phi": [-0.5, 0.5]}
        ]
        
        # Seleccionar región basada en pesos
        weights = [r["weight"] for r in regions]
        region = regions[self.np_random.choice(len(regions), p=weights)]
        
        # Generar coordenadas dentro de la región
        rho = self.np_random.uniform(*region["rho"])
        theta = self.np_random.uniform(*region["theta"])
        phi = self.np_random.uniform(*region["phi"])
        
        # Convertir a cartesianas
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        
        return np.array([x, y, z])


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # --- Curriculum Logic ---
        # Fracción de progreso del curriculum (0 a 1)
        frac = min(self.curriculum_step / self.curriculum_max_steps, 1.0)

        # Rango inicial de ángulos: empieza pequeño y crece hasta los límites completos
        angle_range_low = self.joint_limits_low * (self.initial_angle_range_factor + (1 - self.initial_angle_range_factor) * frac)
        angle_range_high = self.joint_limits_high * (self.initial_angle_range_factor + (1 - self.initial_angle_range_factor) * frac)
        # Asegurarse de que los rangos no se inviertan y respeten los límites absolutos
        angle_range_low = np.clip(angle_range_low, self.joint_limits_low, self.joint_limits_high)
        angle_range_high = np.clip(angle_range_high, self.joint_limits_low, self.joint_limits_high)
        # Evitar rango inválido si low > high
        angle_range_high = np.maximum(angle_range_low, angle_range_high)

        self.joint_angles = self.np_random.uniform(low=angle_range_low, high=angle_range_high).astype(np.float32)
        # Asegurar que los ángulos iniciales estén estrictamente dentro de los límites
        self.joint_angles = np.clip(self.joint_angles, self.joint_limits_low + 1e-6, self.joint_limits_high - 1e-6)

        # --- Target Position ---
        if options and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'], dtype=np.float32)
        else:
            # Generar objetivo aleatorio en un volumen que crece con el curriculum
            min_dist = 0.1 # Distancia mínima desde el hombro
            # Max distancia crece desde un factor inicial hasta el alcance máximo
            max_dist = min_dist + (self.max_reach - min_dist) * (self.initial_target_range_factor + (1 - self.initial_target_range_factor) * frac)
            dist = self.np_random.uniform(min_dist, max(min_dist + 0.01, max_dist))

            # Generar punto en una esfera (distribución uniforme en superficie)
            theta = self.np_random.uniform(0, np.pi) # Ángulo polar (0 a pi)
            phi = self.np_random.uniform(0, 2 * np.pi) # Ángulo azimutal (0 a 2pi)

            # Convertir a cartesianas relativas al hombro
            x_rel = dist * np.sin(theta) * np.cos(phi)
            y_rel = dist * np.sin(theta) * np.sin(phi)
            z_rel = dist * np.cos(theta)

            # Añadir offset del hombro para obtener coordenadas globales (aproximadas)
            # Asumimos que el target se da en el frame base del robot
            self.target_pos = np.array([x_rel, y_rel - self.ShoulderOffsetY, z_rel], dtype=np.float32)

            # Validar si es teóricamente alcanzable (muy simplificado)
            dist_from_shoulder = np.linalg.norm(self.target_pos - np.array([0, -self.ShoulderOffsetY, 0]))
            if dist_from_shoulder > self.max_reach * 1.1: # Dar un margen
                 # Si está muy lejos, regenerar más cerca (podría ser mejor limitar el radio directamente)
                 dist = self.np_random.uniform(min_dist, self.max_reach * 0.8)
                 x_rel = dist * np.sin(theta) * np.cos(phi)
                 y_rel = dist * np.sin(theta) * np.sin(phi)
                 z_rel = dist * np.cos(theta)
                 self.target_pos = np.array([x_rel, y_rel - self.ShoulderOffsetY, z_rel], dtype=np.float32)

        # Calcular estado inicial
        self.end_effector_pos, _ = self._forward_kinematics(self.joint_angles)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
             self._render_frame()

    def _render_frame(self):
        if self.render_mode is None: return

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            limit = self.max_reach * 1.1
            # Ajustar límites para mejor visualización del brazo derecho
            self.ax.set_xlim(-0.1, limit) # X positivo adelante
            self.ax.set_ylim(-limit * 0.8, 0.1) # Y negativo a la derecha
            self.ax.set_zlim(-limit * 0.5, limit * 0.5) # Z arriba/abajo
            self.ax.set_xlabel("X (m) - Adelante")
            self.ax.set_ylabel("Y (m) - Izquierda")
            self.ax.set_zlabel("Z (m) - Arriba")
            self.arm_line, = self.ax.plot([], [], [], 'o-', lw=3, markersize=6, color='cornflowerblue')
            self.target_marker, = self.ax.plot([], [], [], 'gx', markersize=10, mew=3, label='Target')
            self.ef_marker, = self.ax.plot([], [], [], 'ro', markersize=8, label='End Effector')
            self.ax.legend()
            # Establecer vista inicial
            self.ax.view_init(elev=20., azim=-120)


        # Calcular posiciones de todas las articulaciones para dibujar
        j1, j2, j3, j4, j5 = self.joint_angles
        T_base_shoulder = self._get_transform(0, 'x', [0, -self.ShoulderOffsetY, 0])
        T_sp = self._get_transform(j1, 'y')
        T_sr = self._get_transform(j2, 'z')
        T_l1 = self._get_transform(0, 'x', [self.UpperArmLength, 0, 0])
        T_eo = self._get_transform(0, 'x', [0, self.ElbowOffsetY, 0])
        T_ey = self._get_transform(j3, 'z')
        T_er = self._get_transform(j4, 'x')
        T_l2 = self._get_transform(0, 'x', [self.LowerArmLength, 0, 0])
        T_wy = self._get_transform(j5, 'x')

        # Puntos clave
        p_base = np.array([0,0,0]) # Origen global
        p_shoulder_origin = T_base_shoulder[:3, 3] # Punto de anclaje del hombro
        T_shoulder = T_base_shoulder @ T_sp @ T_sr
        # p_shoulder = T_shoulder[:3, 3] # Centro virtual hombro (no existe físicamente)
        T_elbow_frame_origin = T_shoulder @ T_l1 @ T_eo # Origen del frame del codo
        p_elbow = T_elbow_frame_origin[:3, 3]
        T_wrist_frame_origin = T_elbow_frame_origin @ T_ey @ T_er @ T_l2 # Origen del frame de la muñeca
        p_wrist = T_wrist_frame_origin[:3, 3] # Efector final (muñeca)

        # Actualizar datos del plot (Base -> Hombro -> Codo -> Muñeca)
        points = np.array([p_base, p_shoulder_origin, p_elbow, p_wrist])
        self.arm_line.set_data(points[:, 0], points[:, 1])
        self.arm_line.set_3d_properties(points[:, 2])
        self.target_marker.set_data([self.target_pos[0]], [self.target_pos[1]])
        self.target_marker.set_3d_properties([self.target_pos[2]])
        self.ef_marker.set_data([p_wrist[0]], [p_wrist[1]])
        self.ef_marker.set_3d_properties([p_wrist[2]])

        dist = np.linalg.norm(self.target_pos - p_wrist)
        self.ax.set_title(f"Step: {self.current_step}, Dist: {dist:.4f} m, CurrStep: {self.curriculum_step}")

        if self.render_mode == "human":
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(1.0 / self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None
