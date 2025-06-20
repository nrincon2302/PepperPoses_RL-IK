import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

# Importamos directamente las funciones y constantes que necesitamos, sin qibullet
from scripts.CSpace import generate_workspace_points, calculate_joint_positions
from scripts.CSpace import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS, ORIGIN_TORSO
from scripts.CSpace import LEFT_LINK_DISPLACEMENT_VECTOR, RIGHT_LINK_DISPLACEMENT_VECTOR

class PepperAnalyticalEnv(gym.Env):
    """
    Entorno de Gymnasium para control de un brazo de Pepper usando un modelo
    analítico (cinemática directa) en lugar de un simulador físico.
    Esto es extremadamente rápido e ideal para el entrenamiento de IK.
    """
    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(
        self,
        side: str = 'Left',
        max_steps: int = 250,
        n_workspace_samples: int = 8,
    ):
        super().__init__()
        
        # =========================================================
        # Configuración del Entorno Analítico
        # =========================================================
        if side not in ['Left', 'Right']:
            raise ValueError("El parámetro 'side' debe ser 'Left' o 'Right'.")
        self.side = side
        self.joint_limits = LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
        self.link_vectors = LEFT_LINK_DISPLACEMENT_VECTOR if side == 'Left' else RIGHT_LINK_DISPLACEMENT_VECTOR
        
        self.joint_keys = [f'{self.side[0]}ShoulderPitch', f'{self.side[0]}ShoulderRoll', f'{self.side[0]}ElbowYaw', f'{self.side[0]}ElbowRoll', f'{self.side[0]}WristYaw']

        self.joint_limits_low = np.array([self.joint_limits[k][0] for k in self.joint_keys], dtype=np.float32)
        self.joint_limits_high = np.array([self.joint_limits[k][1] for k in self.joint_keys], dtype=np.float32)

        # =========================================================
        # Definición del Espacio de Acciones y Observaciones
        # =========================================================
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.max_steps = max_steps
        self.goal_threshold = 0.02
        
        self.current_step = 0
        self.joint_angles_dict = None
        self.joint_angles_vec = None
        self.current_pos = None
        self.target_pos = None
        self.prev_distance = None

        # =========================================================
        # Espacio de trabajo
        # =========================================================
        points, _ = generate_workspace_points(side=self.side, n_samples=n_workspace_samples)
        self.workspace_points = points.astype(np.float32)
        
        self.max_workspace_radius = np.max(np.linalg.norm(self.workspace_points - np.mean(self.workspace_points, axis=0), axis=1))
        self.current_curriculum_radius = self.max_workspace_radius
        self.np_random, _ = seeding.np_random(None)
    
    # =========================================================
    # Métodos del entorno (simplificados)
    # =========================================================
    def set_curriculum_radius(self, radius: float):
        self.current_curriculum_radius = radius

    def _get_obs(self):
        goal_vector = self.target_pos - self.current_pos
        return np.concatenate([self.joint_angles_vec, goal_vector]).astype(np.float32)
    
    def _get_info(self):
        distance = np.linalg.norm(self.target_pos - self.current_pos)
        return {
            'distance': distance,
            'joint_angles': self.joint_angles_vec.copy(),
            'current_pos': self.current_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'is_success': bool(distance <= self.goal_threshold),
        }

    def _compute_reward(self, distance, prev_distance, action, hit_limits):
        improvement = (prev_distance - distance) * 30.0
        proximity = -2.0 * distance
        smoothness = -0.15 * np.sum(np.square(action))
        limits_penalty = -0.75 if hit_limits else 0.0
        success_bonus = 25.0 if distance <= self.goal_threshold else 0.0
        return improvement + proximity + smoothness + limits_penalty + success_bonus

    def _sample_target(self):
        distances_from_init = np.linalg.norm(self.workspace_points - self.current_pos[None, :], axis=1)
        mask = distances_from_init <= self.current_curriculum_radius
        valid_points = self.workspace_points[mask]
        
        if valid_points.shape[0] == 0:
            closest_idx = np.argmin(distances_from_init)
            valid_points = self.workspace_points[closest_idx][None, :]

        idx = self.np_random.integers(0, len(valid_points))
        base_target = valid_points[idx]
        noise = self.np_random.uniform(-0.02, 0.02, size=3).astype(np.float32)
        return (base_target + noise)

    def is_reachable(self, point: np.ndarray, tol: float = 0.02) -> bool:
        dists = np.linalg.norm(self.workspace_points - point[None, :], axis=1)
        return bool(np.min(dists) <= tol)

    # =========================================================
    # Métodos de Gymnasium
    # =========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Posición inicial aleatoria (vector y diccionario)
        self.joint_angles_vec = self.np_random.uniform(self.joint_limits_low, self.joint_limits_high).astype(np.float32)
        self.joint_angles_dict = {key: self.joint_angles_vec[i] for i, key in enumerate(self.joint_keys)}
        
        # Obtenemos la posición del efector desde el CÁLCULO, no del simulador
        all_joint_positions = calculate_joint_positions(self.side, self.joint_angles_dict)
        self.current_pos = all_joint_positions[-1]

        options = options or {}
        if 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'], dtype=np.float32)
        else:
            self.target_pos = self._sample_target()

        self.prev_distance = np.linalg.norm(self.target_pos - self.current_pos)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32)

        # Aplicar acción y limitar
        new_angles_vec = np.clip(self.joint_angles_vec + action, self.joint_limits_low, self.joint_limits_high)
        hit_limits = np.any(
            (new_angles_vec <= self.joint_limits_low + 1e-6) |
            (new_angles_vec >= self.joint_limits_high - 1e-6)
        )
        
        # Actualizar el diccionario de ángulos para el cálculo
        new_angles_dict = {key: new_angles_vec[i] for i, key in enumerate(self.joint_keys)}
        
        # Obtener nueva posición desde el cálculo, no del simulador
        all_joint_positions = calculate_joint_positions(self.side, new_angles_dict)
        new_pos = all_joint_positions[-1]
        
        distance = np.linalg.norm(self.target_pos - new_pos)
        reward = self._compute_reward(distance, self.prev_distance, action, hit_limits)

        self.joint_angles_vec = new_angles_vec
        self.joint_angles_dict = new_angles_dict
        self.current_pos = new_pos
        self.prev_distance = distance

        terminated = bool(distance <= self.goal_threshold)
        truncated = bool(self.current_step >= self.max_steps)

        observation = self._get_obs()
        info = self._get_info()

        return observation, float(reward), terminated, truncated, info

    def render(self):
        # No hay nada que renderizar
        pass

    def close(self):
        # No hay simulación que cerrar
        pass