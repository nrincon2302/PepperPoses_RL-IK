# --- START OF FILE pepper_env.py ---
# --- MODIFICADO ---

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from qibullet import SimulationManager, PepperVirtual

# Los scripts de CSpace y robot_graph ahora son utilidades, no dependencias del core
from scripts.CSpace import generate_workspace_points
from scripts.robot_graph import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS

class PepperArmEnv(gym.Env):
    """
    Entorno de Gymnasium para control de un brazo de Pepper usando qiBullet para
    la dinámica y la visualización. El currículo se gestiona externamente.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        side: str = 'Left',
        render_mode: str = None, # 'human' inicia la GUI de pybullet
        max_steps: int = 250,
        n_workspace_samples: int = 8,
    ):
        super().__init__()
        
        # --- Configuración del simulador y el robot ---
        self.render_mode = render_mode
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=(self.render_mode == 'human'))
        self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)

        # --- Configuración del brazo y límites articulares ---
        assert side in ('Left', 'Right'), "side debe ser 'Left' o 'Right'"
        self.side = side
        self.joint_limits = LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
        
        # Obtenemos los nombres de los joints de qiBullet
        self.joint_keys = [f'{self.side[0]}ShoulderPitch', f'{self.side[0]}ShoulderRoll', f'{self.side[0]}ElbowYaw', f'{self.side[0]}ElbowRoll', f'{self.side[0]}WristYaw']
        self.end_effector_link_name = f"{self.side[0].lower()}_hand"

        self.joint_limits_low = np.array([self.joint_limits[k][0] for k in self.joint_keys], dtype=np.float32)
        self.joint_limits_high = np.array([self.joint_limits[k][1] for k in self.joint_keys], dtype=np.float32)

        # --- Espacios de acción y observación ---
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # --- Parámetros internos ---
        self.max_steps = max_steps
        self.goal_threshold = 0.05
        
        # --- Estado del episodio ---
        self.current_step = 0
        self.joint_angles = None
        self.current_pos = None
        self.target_pos = None
        self.prev_distance = None

        # --- Workspace y Currículo ---
        points, _ = generate_workspace_points(side=self.side, n_samples=n_workspace_samples)
        self.workspace_points = points.astype(np.float32)
        
        # El radio actual se establece desde fuera a través de un método
        self.max_workspace_radius = np.max(np.linalg.norm(self.workspace_points - np.mean(self.workspace_points, axis=0), axis=1))
        self.current_curriculum_radius = self.max_workspace_radius # Por defecto, sin currículo

        self.np_random, _ = seeding.np_random(None)
    
    def set_curriculum_radius(self, radius: float):
        """Método para que el Callback de entrenamiento actualice el radio."""
        self.current_curriculum_radius = radius

    def _get_obs(self):
        goal_vector = self.target_pos - self.current_pos
        return np.concatenate([self.joint_angles, goal_vector]).astype(np.float32)
    
    def _get_info(self):
        distance = np.linalg.norm(self.target_pos - self.current_pos)
        return {
            'distance': distance,
            'joint_angles': self.joint_angles.copy(),
            'current_pos': self.current_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'is_success': bool(distance <= self.goal_threshold),
        }

    def _compute_reward(self, distance, prev_distance, action, hit_limits):
        improvement = (prev_distance - distance) * 30.0
        proximity = -5.0 * distance
        smoothness = -0.15 * np.sum(np.square(action))
        limits_penalty = -2.0 if hit_limits else 0.0
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Posición inicial aleatoria
        self.joint_angles = self.np_random.uniform(self.joint_limits_low, self.joint_limits_high).astype(np.float32)
        self.pepper.setAngles(self.joint_keys, self.joint_angles.tolist(), 1.0)
        
        # Obtenemos la posición del efector desde el simulador
        link_state = self.pepper.getLinkPosition(self.end_effector_link_name)
        self.current_pos = np.array(link_state[0], dtype=np.float32)

        options = options or {}
        if 'target_pos' in options:
            candidate = np.array(options['target_pos'], dtype=np.float32)
            if not self.is_reachable(candidate):
                raise ValueError(f"Target {candidate} no es alcanzable.")
            self.target_pos = candidate
        else:
            self.target_pos = self._sample_target()

        self.prev_distance = np.linalg.norm(self.target_pos - self.current_pos)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32)

        new_angles = np.clip(self.joint_angles + action, self.joint_limits_low, self.joint_limits_high)
        hit_limits = np.any(
            (new_angles <= self.joint_limits_low + 1e-6) |
            (new_angles >= self.joint_limits_high - 1e-6)
        )
        
        # Aplicar ángulos en el simulador
        self.pepper.setAngles(self.joint_keys, new_angles.tolist(), 1.0)
        
        # Obtener nueva posición desde el simulador
        link_state = self.pepper.getLinkPosition(self.end_effector_link_name)
        new_pos = np.array(link_state[0], dtype=np.float32)
        
        distance = np.linalg.norm(self.target_pos - new_pos)
        reward = self._compute_reward(distance, self.prev_distance, action, hit_limits)

        self.joint_angles = new_angles
        self.current_pos = new_pos
        self.prev_distance = distance

        terminated = bool(distance <= self.goal_threshold)
        truncated = bool(self.current_step >= self.max_steps)

        observation = self._get_obs()
        info = self._get_info()

        return observation, float(reward), terminated, truncated, info

    def render(self):
        # La GUI se actualiza automáticamente si render_mode='human'
        # Esta función es para compatibilidad, pero no hace nada extra
        pass

    def close(self):
        self.simulation_manager.stopSimulation(self.client)