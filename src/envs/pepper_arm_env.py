# src/envs/pepper_arm_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# from scipy.spatial.transform import Rotation as R # No se usa actualmente

from kinematics.forward import forward_kinematics
from curriculum.scheduler import CurriculumScheduler
from memory.history import StateHistory
from utils import normalize_obs #, normalize_angles # normalize_angles no se usa aquí directamente
from rendering.renderer import ArmRenderer # Asumimos que renderer.py existe y funciona

class PepperArmEnv5DOF(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_steps=250, history_size=3):
        super().__init__()
        # Parámetros físicos
        self.params = {
            'shoulder_offset_y': 0.14974, # Distancia del centro del robot al hombro en y
            'upper_arm_length': 0.18120,
            'elbow_offset_y': 0.01500, # Desplazamiento del codo en y local
            'lower_arm_length': 0.15000
        }
        # Límites articulares (radianes)
        self.joint_limits_low = np.array([-2.0857, -1.5620, -2.0857, 0.0087, -1.8239], dtype=np.float32) # RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw
        self.joint_limits_high = np.array([ 2.0857, -0.0087,  2.0857, 1.5620,  1.8239], dtype=np.float32)
        self.num_joints = len(self.joint_limits_low)

        # Módulos auxiliares
        self.scheduler = CurriculumScheduler(num_curriculum_levels=10) # max_steps aquí es para los niveles del currículo
        self.history = StateHistory(size=history_size, dim_joints=self.num_joints, dim_eff_pos=3)
        self.renderer = ArmRenderer(self.params) if render_mode else None

        # Espacios
        action_max = 0.1 # Cambio máximo de ángulo por paso
        self.action_space = spaces.Box(-action_max, action_max, (self.num_joints,), np.float32)
        
        # Dimensión de la observación:
        # current_joint_angles (num_joints)
        # current_end_effector_pos (3)
        # target_pos (3)
        # historical_data ((history_size-1) * (num_joints + 3))
        obs_dim = self.num_joints + 3 + 3 + (self.history.size - 1) * (self.num_joints + 3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

        # Estado interno
        self.joint_angles = np.zeros(self.num_joints, np.float32)
        self.target_pos = np.zeros(3, np.float32)
        self.end_effector_pos = np.zeros(3, np.float32)
        self.prev_distance = None

        # Episodio
        self.max_episode_steps = max_steps
        self.current_step_count = 0 # Renombrado para evitar confusión con self.scheduler.level
        self.goal_threshold = 0.03  # 3 cm
        self.joint_limit_penalty_factor = 0.1 # Penalización más suave

        assert render_mode in (None, *self.metadata['render_modes'])
        self.render_mode = render_mode
        if self.render_mode == "human" and self.renderer:
            self.renderer.init_plot() # Para inicializar la figura de matplotlib una vez

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self.history.clear()

        # Obtener rangos de ángulo según el currículo para el estado inicial
        low_init, high_init = self.scheduler.get_ranges(self.joint_limits_low, self.joint_limits_high)
        self.joint_angles = self.np_random.uniform(low_init, high_init).astype(np.float32)
        
        # Generar o recibir target usando el currículo
        if options and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'], dtype=np.float32)
        else:
            # Usar el currículo para el target también, si se desea
            # Aquí, get_ranges también podría aplicarse a la "dificultad" del target
            # Pero sample_target ya lo hace internamente.
            self.target_pos = self.scheduler.sample_target(
                self.np_random,
                max_reach=self.params['upper_arm_length'] + self.params['lower_arm_length'], # Alcance máximo teórico
                shoulder_offset_y=self.params['shoulder_offset_y']
            )
        
        self.end_effector_pos, _ = forward_kinematics(self.joint_angles, self.params)
        self.prev_distance = np.linalg.norm(self.target_pos - self.end_effector_pos)

        # Llenar el historial con el estado inicial repetido para que tenga el tamaño correcto
        for _ in range(self.history.size):
            self.history.append(self.joint_angles, self.end_effector_pos)
            
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human' and self.renderer:
            self.renderer.render(self.joint_angles, self.target_pos, self.current_step_count, info['curriculum_level'])
        return obs, info

    def step(self, action):
        prev_dist_to_target = self.prev_distance
        
        # Aplicar acción
        current_angles_before_action = self.joint_angles.copy()
        self.joint_angles = self.joint_angles + np.array(action, dtype=np.float32)
        
        # Clipping y cálculo de penalización por límite articular
        violated_limits = np.logical_or(self.joint_angles < self.joint_limits_low, 
                                        self.joint_angles > self.joint_limits_high)
        joint_limit_penalty = np.sum(violated_limits) * self.joint_limit_penalty_factor
        
        self.joint_angles = np.clip(self.joint_angles, self.joint_limits_low, self.joint_limits_high)
        
        # Actualizar posición del efector final
        self.end_effector_pos, _ = forward_kinematics(self.joint_angles, self.params)
        
        # Actualizar historial con el *nuevo* estado
        self.history.append(self.joint_angles, self.end_effector_pos)

        new_dist_to_target = np.linalg.norm(self.target_pos - self.end_effector_pos)
        self.prev_distance = new_dist_to_target # Actualizar para el siguiente paso

        # Recompensa
        reward = self._compute_reward(new_dist_to_target, prev_dist_to_target, action, joint_limit_penalty)
        
        # Terminación y truncamiento
        terminated = new_dist_to_target <= self.goal_threshold
        self.current_step_count += 1
        truncated = self.current_step_count >= self.max_episode_steps

        # Observación e info
        obs = self._get_obs()
        info = self._get_info()
        info["is_success"] = terminated # SB3 Monitor usa esto

        if self.render_mode in ('human', 'rgb_array') and self.renderer:
            self.renderer.render(self.joint_angles, self.target_pos,
                                  self.current_step_count, info['curriculum_level'])
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        historical_data_concat = self.history.get_concatenated_history()
        return normalize_obs(self.joint_angles, self.end_effector_pos, self.target_pos, historical_data_concat)

    def _compute_reward(self, dist, prev_dist, action, joint_limit_penalty_val):
        # Recompensa por acercarse (shaping)
        reward_distance_improvement = (prev_dist - dist) * 20.0 
        
        # Recompensa por estar cerca (sparse-like, pero continua)
        reward_proximity = 10.0 / (1.0 + 20.0 * dist**2) # Más agresiva cerca del objetivo
        
        # Penalización por energía/esfuerzo
        reward_action_magnitude = -0.1 * np.sum(np.square(action)) # Aumentada ligeramente
        
        # Penalización por tocar límites
        reward_joint_limits = -joint_limit_penalty_val
        
        # Bonus por alcanzar el objetivo
        reward_goal_achieved = 0.0
        if dist <= self.goal_threshold:
            reward_goal_achieved = 50.0
            
        total_reward = (reward_distance_improvement + 
                        reward_proximity + 
                        reward_action_magnitude + 
                        reward_joint_limits + 
                        reward_goal_achieved)
        return total_reward

    def _get_info(self):
        return {
            'distance': self.prev_distance,
            'curriculum_level': self.scheduler.level,
            'target_pos': self.target_pos.copy(),
            'end_effector_pos': self.end_effector_pos.copy()
        }

    def render(self):
        if self.renderer:
            if self.render_mode == 'rgb_array':
                return self.renderer.render(self.joint_angles, self.target_pos,
                                            self.current_step_count, self.scheduler.level, get_image=True)
            elif self.render_mode == 'human':
                self.renderer.render(self.joint_angles, self.target_pos,
                                      self.current_step_count, self.scheduler.level)
        # Si no hay renderer o el modo no es rgb_array, no se retorna imagen explícitamente.
        # Gymnasium espera que render() devuelva un np.ndarray para 'rgb_array'.

    def close(self):
        if self.renderer:
            self.renderer.close()