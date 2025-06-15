# --- START OF FILE pepper_env.py ---
# --- VERSIÓN FINAL OPTIMIZADA ---

import time
import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from qibullet import SimulationManager

# Los scripts de CSpace y robot_graph ahora son utilidades, no dependencias del core
from scripts.CSpace import generate_workspace_points
from scripts.robot_graph import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS

class PepperArmEnv(gym.Env):
    """
    Entorno de Gymnasium de alto rendimiento para control de un brazo de Pepper.
    Utiliza control de bajo nivel de PyBullet para una simulación rápida, ideal para RL.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        side: str = 'Left',
        render_mode: str = None,
        max_steps: int = 250,
        n_workspace_samples: int = 8,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.side = side
        self.max_steps = max_steps
        self.goal_threshold = 0.10

        # --- Configuración del simulador y el robot ---
        self.simulation_manager = SimulationManager()
        connection_mode = p.GUI if self.render_mode == 'human' else p.DIRECT
        self.client = self.simulation_manager.launchSimulation(gui=True if connection_mode==p.GUI else False)
        
        self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)
        self.pepper_id = self.pepper.getRobotModel()

        # --- Frecuencias de simulación y control ---
        self.sim_frequency = 240  # Hz
        self.control_frequency = 30  # Hz
        self.sim_steps_per_control_step = self.sim_frequency // self.control_frequency
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.sim_frequency, physicsClientId=self.client)

        # --- Configuración del brazo y límites articulares ---
        assert side in ('Left', 'Right'), "side debe ser 'Left' o 'Right'"
        self.joint_limits = LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
        
        self.joint_keys = [f'{self.side[0]}ShoulderPitch', f'{self.side[0]}ShoulderRoll', f'{self.side[0]}ElbowYaw', f'{self.side[0]}ElbowRoll', f'{self.side[0]}WristYaw']
        self.end_effector_link_name = f"{self.side.lower()}_hand"
        
        all_joint_names = [p.getJointInfo(self.pepper_id, i)[1].decode('UTF-8') for i in range(p.getNumJoints(self.pepper_id))]
        self.joint_indices = [all_joint_names.index(name) for name in self.joint_keys]

        self.joint_limits_low = np.array([self.joint_limits[k][0] for k in self.joint_keys], dtype=np.float32)
        self.joint_limits_high = np.array([self.joint_limits[k][1] for k in self.joint_keys], dtype=np.float32)
        self.joint_centers = (self.joint_limits_low + self.joint_limits_high) / 2.0
        self.joint_ranges = self.joint_limits_high - self.joint_limits_low

        # --- Espacios de acción y observación ---
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32) # 5 angs + 5 vels + 3 goal_vec

        # --- Workspace y Currículo ---
        points, _ = generate_workspace_points(side=self.side, n_samples=n_workspace_samples)
        self.workspace_points = points.astype(np.float32)
        self.max_workspace_radius = np.max(np.linalg.norm(self.workspace_points - np.mean(self.workspace_points, axis=0), axis=1))
        self.current_curriculum_radius = self.max_workspace_radius

        # --- Estado del episodio ---
        self.current_step = 0
        self.joint_angles = None
        self.joint_velocities = None
        self.current_pos = None
        self.target_pos = None
        self.prev_distance = None

        self.np_random, _ = np.random.RandomState(None), None
    
    def seed(self, seed=None):
        """Para compatibilidad con API antigua de gym."""
        self.np_random, seed_val = np.random.RandomState(seed), seed
        return [seed_val]

    def set_curriculum_radius(self, radius: float):
        self.current_curriculum_radius = radius

    def _update_robot_state(self):
        """Lee y actualiza el estado completo del robot desde PyBullet."""
        joint_states = p.getJointStates(self.pepper_id, self.joint_indices, physicsClientId=self.client)
        self.joint_angles = np.array([state[0] for state in joint_states], dtype=np.float32)
        self.joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)
        
        # Posición del efector final
        link_state = p.getLinkState(self.pepper_id, self.joint_indices[-1] + 1, computeForwardKinematics=True, physicsClientId=self.client)
        self.current_pos = np.array(link_state[0], dtype=np.float32)
    
    def _get_obs(self):
        goal_vector = self.target_pos - self.current_pos
        return np.concatenate([self.joint_angles, self.joint_velocities, goal_vector]).astype(np.float32)
    
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
        # 1. Penalización por distancia (potencial negativo)
        distance_penalty = -5.0 * distance
        # 2. Recompensa por mejora (shaping)
        improvement_reward = (prev_distance - distance) * 30.0
        # 3. Penalización por esfuerzo (acciones suaves)
        action_penalty = -0.15 * np.sum(np.square(action))
        # 4. Penalización por chocar con los límites
        limits_penalty = -0.5 if hit_limits else 0.0
        # 5. Bonus por éxito (reducido)
        success_bonus = 25.0 if distance <= self.goal_threshold else 0.0
        # 6. Penalización anti-oscilación
        settling_penalty = 0.0
        if distance < (self.goal_threshold * 1.5):
            speed = np.linalg.norm(self.joint_velocities, ord=1)
            settling_penalty = -0.5 * speed
        # 7. Bonus de exploración articular
        normalized_dist_from_center = np.abs(self.joint_angles - self.joint_centers) / (self.joint_ranges / 2.0)
        exploration_bonus = 0.05 * np.sum(normalized_dist_from_center)
        
        return (distance_penalty + improvement_reward + action_penalty + limits_penalty +
                success_bonus + settling_penalty + exploration_bonus)

    def _sample_target(self):
        distances_from_init = np.linalg.norm(self.workspace_points - self.current_pos[None, :], axis=1)
        mask = distances_from_init <= self.current_curriculum_radius
        valid_points = self.workspace_points[mask]
        
        if valid_points.shape[0] == 0:
            closest_idx = np.argmin(distances_from_init)
            valid_points = self.workspace_points[closest_idx][None, :]

        idx = self.np_random.randint(0, len(valid_points))
        base_target = valid_points[idx]
        noise = self.np_random.uniform(-0.02, 0.02, size=3).astype(np.float32)
        return (base_target + noise)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        self.current_step = 0
        options = options or {}

        # Resetear el estado del robot a una pose aleatoria
        initial_angles = self.np_random.uniform(self.joint_limits_low, self.joint_limits_high)
        for i, angle in enumerate(initial_angles):
            p.resetJointState(self.pepper_id, self.joint_indices[i], angle, physicsClientId=self.client)
        
        self._update_robot_state()

        if 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'], dtype=np.float32)
        else:
            self.target_pos = self._sample_target()

        self.prev_distance = np.linalg.norm(self.target_pos - self.current_pos)
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        
        target_angles = np.clip(self.joint_angles + action, self.joint_limits_low, self.joint_limits_high)
        hit_limits = np.any((target_angles <= self.joint_limits_low + 1e-6) | (target_angles >= self.joint_limits_high - 1e-6))
        
        p.setJointMotorControlArray(
            bodyUniqueId=self.pepper_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=[50.0] * len(self.joint_indices),
            positionGains=[0.03] * len(self.joint_indices),
            velocityGains=[1.0] * len(self.joint_indices),
            physicsClientId=self.client
        )
        
        for _ in range(self.sim_steps_per_control_step):
            p.stepSimulation(physicsClientId=self.client)
            if self.render_mode == 'human':
                time.sleep(1.0 / self.sim_frequency)

        self._update_robot_state()
        
        distance = np.linalg.norm(self.target_pos - self.current_pos)
        reward = self._compute_reward(distance, self.prev_distance, action, hit_limits)
        
        self.prev_distance = distance
        
        terminated = bool(distance <= self.goal_threshold)
        truncated = bool(self.current_step >= self.max_steps)

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def close(self):
        if self.client is not None:
            self.simulation_manager.stopSimulation(self.client)
            self.client = None