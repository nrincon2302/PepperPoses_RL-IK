import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from qibullet import SimulationManager, PepperVirtual

from scripts.CSpace import generate_workspace_points, LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS

class PepperArmEnv(gym.Env):
    """
    Entorno de Gymnasium para control de un brazo de Pepper usando qiBullet para
    la dinámica y la visualización. El currículo se gestiona externamente.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        side: str = 'Left',
        render_mode: str = None, # Ponemos 'None' por defecto para que no abra GUI si no se especifica
        max_steps: int = 250,
        n_workspace_samples: int = 8,
    ):
        super().__init__()
        
        # =========================================================
        # Configuración del Simulador y el Entorno
        # =========================================================
        self.render_mode = render_mode
        self.simulation_manager = SimulationManager()
        # El GUI se activa solo si el modo de render es 'human'
        self.client = self.simulation_manager.launchSimulation(gui=(self.render_mode == 'human'))
        self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)

        # Configuración del brazo con sus límites de joints
        if side not in ['Left', 'Right']:
            raise ValueError("El parámetro 'side' debe ser 'Left' o 'Right'.")
        self.side = side
        self.joint_limits = LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
        
        # Obtenemos los nombres de los joints de qiBullet
        self.joint_keys = [f'{self.side[0]}ShoulderPitch', f'{self.side[0]}ShoulderRoll', f'{self.side[0]}ElbowYaw', f'{self.side[0]}ElbowRoll', f'{self.side[0]}WristYaw']
        self.end_effector_link_name = f"{self.side[0].lower()}_hand"

        self.joint_limits_low = np.array([self.joint_limits[k][0] for k in self.joint_keys], dtype=np.float32)
        self.joint_limits_high = np.array([self.joint_limits[k][1] for k in self.joint_keys], dtype=np.float32)

        # =========================================================
        # Definición del Espacio de Acciones y Observaciones
        # =========================================================
        # Espacio de Acción: Codifica los deltas de los ángulos de los joints
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(5,), dtype=np.float32)
        # Espacio de Observación: Ángulos de los joints + vector hacia el objetivo
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # Máxima cantidad de pasos por episodio y umbral de éxito (error en centímetros convertido a metros)
        self.max_steps = max_steps
        self.goal_threshold = 0.10
        
        # Estado de un episodio
        self.current_step = 0
        self.joint_angles = None
        self.current_pos = None
        self.target_pos = None
        self.prev_distance = None

        # =========================================================
        # Espacio de trabajo y Configuración del currículo
        # =========================================================
        # Generación de puntos del espacio de trabajo (ahora usa caché)
        points, _ = generate_workspace_points(side=self.side, n_samples=n_workspace_samples)
        self.workspace_points = points.astype(np.float32)
        
        # El radio se calcula según la etapa del currículo y se emplea sampling aleatorio para los objetivos
        self.max_workspace_radius = np.max(np.linalg.norm(self.workspace_points - np.mean(self.workspace_points, axis=0), axis=1))
        self.current_curriculum_radius = self.max_workspace_radius
        self.np_random, _ = seeding.np_random(None)
    

    # =========================================================
    # Métodos del entorno modelado
    # =========================================================
    def set_curriculum_radius(self, radius: float):
        """
        Método para que el Callback de entrenamiento actualice el radio.
        Permite ajustar el radio del currículo dinámicamente.
        """
        self.current_curriculum_radius = radius

    def _get_obs(self):
        """
        Obtiene la observación actual del entorno calculando el vector de error
        y concatenando los ángulos de cada joint con dicho vector de error.
        """
        goal_vector = self.target_pos - self.current_pos
        return np.concatenate([self.joint_angles, goal_vector]).astype(np.float32)
    
    def _get_info(self):
        """
        Obtiene información adicional del entorno, como la distancia al objetivo,
        los ángulos de los joints, la posición actual y la posición del objetivo.
        También indica si se ha alcanzado el objetivo.
        """
        distance = np.linalg.norm(self.target_pos - self.current_pos)
        return {
            'distance': distance,
            'joint_angles': self.joint_angles.copy(),
            'current_pos': self.current_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'is_success': bool(distance <= self.goal_threshold),
        }

    def _compute_reward(self, distance, prev_distance, action, hit_limits):
        """
        Calcula la recompensa basada en la distancia al objetivo, el cambio en la distancia,
        la suavidad del movimiento (acción), si se han alcanzado los límites de los joints
        y si se ha alcanzado el objetivo.
        """
        # Recompensa debido a mejoramiento en la distancia
        improvement = (prev_distance - distance) * 30.0
        # Penalización por falta de proximidad al objetivo
        proximity = -2.0 * distance
        # Penalización por acción brusca (movimiento poco suave de los joints)
        smoothness = -0.15 * np.sum(np.square(action))
        # Penalización por alcanzar los límites de los joints o moverse fuera de estos
        limits_penalty = -0.75 if hit_limits else 0.0
        # Bonificación por alcanzar el objetivo
        success_bonus = 25.0 if distance <= self.goal_threshold else 0.0

        # Recompensa total combinando todos los componentes
        return improvement + proximity + smoothness + limits_penalty + success_bonus

    def _sample_target(self):
        """
        Muestra un objetivo aleatorio dentro del espacio de trabajo, asegurándose de que
        esté dentro del radio actual del currículo. Si no hay puntos válidos, selecciona
        el más cercano al punto inicial.
        """
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
        """
        Verifica si un punto dado es alcanzable por el brazo de Pepper.
        Compara la distancia del punto a los puntos del espacio de trabajo.
        Si la distancia mínima es menor o igual a la tolerancia, se considera alcanzable.
        """
        dists = np.linalg.norm(self.workspace_points - point[None, :], axis=1)
        return bool(np.min(dists) <= tol)


    # =========================================================
    # Métodos de Gymnasium
    # =========================================================
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al final de un episodio, configurando los ángulos 
        de los joints a una posición aleatoria dentro de los límites definidos y
        estableciendo un objetivo aleatorio dentro del espacio de trabajo.
        También inicializa el paso actual y calcula la posición del efector final.
        """
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
        """
        Ejecuta un paso en el entorno, aplicando la acción al brazo de Pepper,
        actualizando los ángulos de los joints, obteniendo la nueva posición del efector final,
        calculando la recompensa y verificando si se ha alcanzado el objetivo o si se ha agotado
        el número máximo de pasos.
        """
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
        pass

    def close(self):
        self.simulation_manager.stopSimulation(self.client)
