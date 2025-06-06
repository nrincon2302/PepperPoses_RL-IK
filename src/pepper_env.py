# File: pepper_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scripts.CSpace import generate_workspace_points
from scripts.robot_graph import (
    LEFT_JOINT_LIMITS,
    RIGHT_JOINT_LIMITS,
    calculate_joint_positions,
    plot_robot
)

from scipy.spatial import cKDTree


class PepperArmEnv(gym.Env):
    """
    Entorno de Gymnasium para control de un brazo (Left/Right) de Pepper.
    Integra un esquema de Curriculum Learning que requiere 5 éxitos consecutivos
    para incrementar el nivel de dificultad (el radio curricular).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        side: str = 'Left',
        render_mode: str = None,
        max_steps: int = 250,
        n_workspace_samples: int = 8,
        curriculum_start_frac: float = 0.2,
        curriculum_increment_frac: float = 0.1,
        required_consecutive_successes: int = 5
    ):
        """
        Args:
            side (str): 'Left' o 'Right' para controlar el brazo correspondiente.
            render_mode (str): 'human' o 'rgb_array'. Si es 'rgb_array', retorna array de píxeles
                               (no implementado aquí, pero placeholder).
            max_steps (int): Máximo de pasos por episodio antes de truncar.
            n_workspace_samples (int): Cantidad de muestras por dimensión para generar puntos alcanzables.
            curriculum_start_frac (float): Fracción inicial del radio curricular (entre 0 y 1).
            curriculum_increment_frac (float): Fracción del radio total que se añade tras 5 éxitos.
            required_consecutive_successes (int): Número de éxitos consecutivos para subir radio curricular.
        """
        super().__init__()

        # --------- Configuración de brazo y límites articulares ---------
        assert side in ('Left', 'Right'), "side debe ser 'Left' o 'Right'"
        self.side = side
        self.joint_limits = (
            LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
        )
        self.joint_keys = list(self.joint_limits.keys())

        # Vectores (bajo, alto) en radianes
        self.joint_limits_low = np.array(
            [self.joint_limits[k][0] for k in self.joint_keys], dtype=np.float32
        )
        self.joint_limits_high = np.array(
            [self.joint_limits[k][1] for k in self.joint_keys], dtype=np.float32
        )

        # --------- Espacios de acción y observación ---------
        # Acción: delta de 5 ángulos, cada uno ∈ [–0.05, +0.05]
        self.action_space = spaces.Box(
            low=-0.05,
            high=0.05,
            shape=(5,),
            dtype=np.float32
        )

        # Observación: [5 ángulos, 3 posición actual, 3 posición objetivo] = 11 dimensiones
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

        # --------- Parámetros internos ---------
        self.max_steps = max_steps
        self.current_step = 0
        self.goal_threshold = 0.02  # 2 cm de tolerancia

        self.joint_angles = None     # Array numpy (5,)
        self.current_pos = None      # Array numpy (3,) → [x, y, z]
        self.target_pos = None       # Array numpy (3,)
        self.prev_distance = None    # Distancia al objetivo en paso anterior

        self.render_mode = render_mode

        # --------- Curriculum Learning Setup ---------
        # 1) Generar nube de puntos alcanzables
        points, _ = generate_workspace_points(
            side=self.side,
            n_samples=n_workspace_samples
        )
        self.workspace_points = points.astype(np.float32)

        # 2) Construir KDTree para queries de factibilidad
        self._kdtree = cKDTree(self.workspace_points)

        # 3) Calcular posición efector en pose neutra (todos ángulos = 0)
        zero_angles = {k: 0.0 for k in self.joint_keys}
        zero_positions = calculate_joint_positions(
            side=self.side,
            joint_angles=zero_angles
        )
        self.zero_pos = zero_positions[-1].astype(np.float32)

        # 4) Distancias de cada punto a zero_pos, y radio máximo
        self.distances_to_zero = np.linalg.norm(
            self.workspace_points - self.zero_pos[None, :],
            axis=1
        )
        self.max_distance = float(np.max(self.distances_to_zero))

        # 5) Parámetros de currículo
        self.curriculum_start_frac = curriculum_start_frac
        self.curriculum_increment_frac = curriculum_increment_frac
        self.curriculum_radius = curriculum_start_frac * self.max_distance
        self.curriculum_increment = curriculum_increment_frac * self.max_distance

        # 6) Éxitos consecutivos necesarios para subir nivel
        self.required_consecutive_successes = required_consecutive_successes
        self.success_consecutive = 0  # contador de éxitos consecutivos
        self.episode_count = 0        # conteo total de episodios exitosos (para referencia)

        # Inicializar estado del episodio
        self._init_episode_state()

    def _init_episode_state(self):
        """Variables internas que reiniciamos en cada reset()"""
        self.current_step = 0
        self.joint_angles = None
        self.current_pos = None
        self.target_pos = None
        self.prev_distance = None

    def _compute_reward(self, distance, prev_distance, action, hit_limits):
        """
        Calcula recompensa combinando:
          - Mejora en distancia (prev_distance – distance) × 30.0
          - Proximidad al objetivo: 15.0 / (1 + 25 × distance²)
          - Penalización por movimientos bruscos: –0.15 × ‖action‖²
          - Penalización por hit_limits: –0.4 si true, else 0
          - Bonus de éxito: +100 si distance <= goal_threshold
        """
        improvement = (prev_distance - distance) * 30.0 if prev_distance is not None else 0.0
        proximity = 15.0 / (1.0 + 25.0 * distance**2)
        smoothness = -0.15 * np.sum(np.square(action))
        limits_penalty = -0.4 if hit_limits else 0.0
        success_bonus = 100.0 if distance <= self.goal_threshold else 0.0
        return improvement + proximity + smoothness + limits_penalty + success_bonus

    def _update_curriculum(self):
        """
        Incrementa el radio curricular en curriculum_increment, sin exceder max_distance,
        y resetea el contador de éxitos consecutivos (porque acabamos de subir nivel).
        """
        self.episode_count += 1
        self.curriculum_radius = min(
            self.curriculum_radius + self.curriculum_increment,
            self.max_distance
        )
        # Una vez subimos de nivel, reseteamos éxitos consecutivos
        self.success_consecutive = 0

    def _sample_target(self):
        """
        Retorna un punto meta (x,y,z) cuya distancia a self.current_pos sea ≤ self.curriculum_radius:
          - Medir distancias desde la posición inicial (current_pos) a cada workspace_point
          - Filtrar por curriculu_radius
          - Si no hay ninguno, usar todo workspace_points
          - Tomar uno al azar y añadir ruido uniforme en [-0.02, +0.02]
        """
        # Calcular distancias desde la posición inicial
        distances_from_init = np.linalg.norm(
            self.workspace_points - self.current_pos[None, :],
            axis=1
        )

        mask = (distances_from_init <= self.curriculum_radius)
        valid_points = self.workspace_points[mask]
        if valid_points.shape[0] == 0:
            valid_points = self.workspace_points

        idx = self.np_random.integers(0, len(valid_points))
        base = valid_points[idx]
        ruido = self.np_random.uniform(-0.02, 0.02, size=3).astype(np.float32)
        return (base + ruido).astype(np.float32)

    def is_reachable(self, point: np.ndarray, tol: float = 0.02) -> bool:
        """
        Verifica si 'point' está en el workspace alcanzable (dentro de tol metros
        de algún punto muestreado).
        """
        dist, _ = self._kdtree.query(point)
        return (dist <= tol)

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno para un nuevo episodio.
        Genera ángulos aleatorios, calcula posición inicial y muestrea meta
        según el radio curricular actual (distancia desde la posición inicial).
        Si options['target_pos'] está definida, valida y usa esa posición exacta.
        Devuelve (observation, info).
        """
        super().reset(seed=seed)
        self._init_episode_state()

        # 1) Ángulos iniciales aleatorios dentro de límites articulares
        self.joint_angles = self.np_random.uniform(
            self.joint_limits_low,
            self.joint_limits_high
        ).astype(np.float32)

        # 2) Calcular posición actual del efector con esos ángulos
        joint_dict = dict(zip(self.joint_keys, self.joint_angles))
        positions = calculate_joint_positions(
            side=self.side,
            joint_angles=joint_dict
        )
        self.current_pos = positions[-1].astype(np.float32)

        # 3) Seleccionar target_pos
        if options and ('target_pos' in options):
            candidate = np.array(options['target_pos'], dtype=np.float32)
            if not self.is_reachable(candidate):
                raise ValueError(f"Target {candidate} no alcanzable.")
            self.target_pos = candidate
        else:
            self.target_pos = self._sample_target()

        # 4) Calcular distancia inicial
        self.prev_distance = np.linalg.norm(self.target_pos - self.current_pos)

        # 5) Construir observación e info
        observation = np.concatenate([
            self.joint_angles,
            self.current_pos,
            self.target_pos
        ]).astype(np.float32)

        info = {
            'distance': float(self.prev_distance),
            'joint_angles': self.joint_angles.copy()
        }
        return observation, info

    def step(self, action):
        """
        Aplica la acción (delta ángulos).
        Retorna (observation, reward, terminated, truncated, info).
        En caso de éxito, incrementa contador de éxitos consecutivos.
        Si se cumplen 5 éxitos consecutivos, sube de nivel (radio curricular).
        """
        self.current_step += 1

        # 1) Aplicar acción y clip a límites articulares
        new_angles = self.joint_angles + action.astype(np.float32)
        new_angles = np.clip(new_angles, self.joint_limits_low, self.joint_limits_high)

        # 2) Calcular nueva posición del efector
        joint_dict = dict(zip(self.joint_keys, new_angles))
        positions = calculate_joint_positions(
            side=self.side,
            joint_angles=joint_dict
        )
        new_pos = positions[-1].astype(np.float32)

        # 3) Calcular distancia y si golpea límites
        distance = np.linalg.norm(self.target_pos - new_pos)
        hit_limits = np.any(
            (new_angles <= self.joint_limits_low + 1e-6) |
            (new_angles >= self.joint_limits_high - 1e-6)
        )

        # 4) Recompensa
        reward = self._compute_reward(
            distance=distance,
            prev_distance=self.prev_distance,
            action=action,
            hit_limits=hit_limits
        )

        # 5) Actualizar estado interno
        self.joint_angles = new_angles
        self.current_pos = new_pos
        self.prev_distance = distance

        # 6) Determinar terminado/truncado
        terminated = bool(distance <= self.goal_threshold)
        truncated = bool(self.current_step >= self.max_steps)

        # 7) Manejo de éxitos consecutivos y currículo
        if terminated:
            self.success_consecutive += 1
            # Solo al alcanzar 5 éxitos consecutivos subimos nivel
            if self.success_consecutive >= self.required_consecutive_successes:
                self._update_curriculum()
        else:
            # Si falla, resetear contador de éxitos consecutivos
            self.success_consecutive = 0

        # 8) Construir observación e info a devolver
        observation = np.concatenate([
            self.joint_angles,
            self.current_pos,
            self.target_pos
        ]).astype(np.float32)
        info = {
            'distance': float(distance),
            'joint_angles': self.joint_angles.copy(),
            'is_success': terminated,
            'curriculum_radius': float(self.curriculum_radius),
            'success_consecutive': int(self.success_consecutive)
        }
        return observation, float(reward), terminated, truncated, info

    def render(self):
        """
        Renderiza el brazo de Pepper en la pose actual, usando Matplotlib 3D.
        Solo válido si render_mode == 'human'.
        """
        if self.render_mode == "human":
            angles_dict = dict(zip(self.joint_keys, self.joint_angles))
            if self.side == 'Left':
                plot_robot(left_angles=angles_dict, right_angles=None)
            else:
                plot_robot(left_angles=None, right_angles=angles_dict)

    def close(self):
        """No hace nada (placeholder para compatibilidad)."""
        pass


if __name__ == "__main__":
    """
    Bloque interactivo para probar el entorno desde consola.
    Se pide al usuario parámetros básicos y se ejecuta un episodio con acciones aleatorias.
    """
    print("\n=== Prueba interactiva de PepperArmEnv ===")

    # --------- Pedir parámetros al usuario ---------
    side = input("¿Qué brazo entrenar? (Left/Right) [Left]: ").strip() or "Left"
    if side not in ("Left", "Right"):
        print("Entrada inválida, se usará 'Left'.")
        side = "Left"

    try:
        n_samples = int(input("n_workspace_samples (número de muestras por dimensión) [8]: ") or "8")
    except ValueError:
        print("No es un número válido. Se usará 8.")
        n_samples = 8

    try:
        start_frac = float(input("curriculum_start_frac (ej. 0.2) [0.2]: ") or "0.2")
    except ValueError:
        print("No es un float válido. Se usará 0.2.")
        start_frac = 0.2

    try:
        incr_frac = float(input("curriculum_increment_frac (ej. 0.1) [0.1]: ") or "0.1")
    except ValueError:
        print("No es un float válido. Se usará 0.1.")
        incr_frac = 0.1

    try:
        max_steps = int(input("max_steps por episodio [250]: ") or "250")
    except ValueError:
        print("No es un número válido. Se usará 250.")
        max_steps = 250

    try:
        req_succ = int(input("Éxitos consecutivos para subir nivel [5]: ") or "5")
    except ValueError:
        print("No es un número válido. Se usarán 5 éxitos consecutivos.")
        req_succ = 5

    print(f"\nCreando entorno PepperArmEnv(side='{side}', n_workspace_samples={n_samples}, "
          f"start_frac={start_frac}, incr_frac={incr_frac}, max_steps={max_steps}, "
          f"required_successes={req_succ})...\n")

    env = PepperArmEnv(
        side=side,
        render_mode="human",
        max_steps=max_steps,
        n_workspace_samples=n_samples,
        curriculum_start_frac=start_frac,
        curriculum_increment_frac=incr_frac,
        required_consecutive_successes=req_succ
    )

    # --------- Ejecutar un solo episodio con acciones aleatorias ---------
    obs, info = env.reset()
    print(f"Estado inicial:\n  Ángulos iniciales: {info['joint_angles']}\n"
          f"  Posición inicial: {obs[5:8]}\n"
          f"  Meta inicial (x,y,z): {obs[8:11]}\n"
          f"  Distancia inicial: {info['distance']:.4f} m\n"
          f"  Radio curricular: {info.get('curriculum_radius', env.curriculum_radius):.4f} m\n")

    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info_step = env.step(action)
        total_reward += reward
        print(f"Step {env.current_step:03d} → distancia={info_step['distance']:.4f} m, "
              f"reward={reward:.2f}, success={info_step['is_success']}, "
              f"consec_success={info_step['success_consecutive']}, "
              f"curric_radius={info_step['curriculum_radius']:.4f}")
        env.render()

        if terminated or truncated:
            done = True

    if info_step['is_success']:
        print(f"\n¡Éxito! El brazo alcanzó la meta. Recompensa total={total_reward:.2f}")
    else:
        print(f"\nSe truncó el episodio tras {env.current_step} pasos. Distancia final={info_step['distance']:.4f} m")

    print("\nFin de la prueba interactiva.\n")
