import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from scripts.CSpace import generate_workspace_points
from scripts.robot_graph import (
    LEFT_JOINT_LIMITS,
    RIGHT_JOINT_LIMITS,
    calculate_joint_positions,
    plot_robot
)


class PepperArmEnv(gym.Env):
    """
    Entorno de Gymnasium para control de un brazo (Left/Right) de Pepper.
    Integra Curriculum Learning que requiere 5 éxitos consecutivos
    para incrementar el radio curricular. Incluye el método seed().
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
            side (str): 'Left' o 'Right' para seleccionar el brazo.
            render_mode (str): 'human' o 'rgb_array'. (rgb_array no implementado aquí).
            max_steps (int): tope de pasos por episodio.
            n_workspace_samples (int): número de muestras por dimensión para el workspace.
            curriculum_start_frac (float): fracción inicial del radio curricular (0 < frac ≤ 1).
            curriculum_increment_frac (float): fracción del radio total para incrementar.
            required_consecutive_successes (int): éxitos seguidos para subir nivel.
        """
        super().__init__()

        # --- Configuración del brazo y límites articulares ---
        assert side in ('Left', 'Right'), "side debe ser 'Left' o 'Right'"
        self.side = side
        self.joint_limits = (
            LEFT_JOINT_LIMITS if side == 'Left' else RIGHT_JOINT_LIMITS
        )
        self.joint_keys = list(self.joint_limits.keys())

        # Vectores (low, high) en radianes
        self.joint_limits_low = np.array(
            [self.joint_limits[k][0] for k in self.joint_keys], dtype=np.float32
        )
        self.joint_limits_high = np.array(
            [self.joint_limits[k][1] for k in self.joint_keys], dtype=np.float32
        )

        # --- Espacios de acción y observación ---
        # Acción: delta de 5 ángulos ∈ [–0.05, +0.05]
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

        # --- Parámetros internos ---
        self.max_steps = max_steps
        self.current_step = 0
        self.goal_threshold = 0.10  # 10 cm

        self.joint_angles = None   # numpy (5,)
        self.current_pos = None    # numpy (3,)
        self.target_pos = None     # numpy (3,)
        self.prev_distance = None  # float

        self.render_mode = render_mode

        # --- Curriculum Learning Setup ---
        # 1) Generar puntos alcanzables
        points, _ = generate_workspace_points(
            side=self.side,
            n_samples=n_workspace_samples
        )
        self.workspace_points = points.astype(np.float32)  # (N,3)

        # 2) Calcular la posición del efector en pose neutra (todos ceros)
        zero_angles = {k: 0.0 for k in self.joint_keys}
        zero_positions = calculate_joint_positions(
            side=self.side,
            joint_angles=zero_angles
        )
        self.zero_pos = zero_positions[-1].astype(np.float32)

        # 3) Distancias de cada punto a zero_pos, y radio máximo
        self.distances_to_zero = np.linalg.norm(
            self.workspace_points - self.zero_pos[None, :],
            axis=1
        )
        self.max_distance = float(np.max(self.distances_to_zero))

        # 4) Parámetros de currículo
        self.curriculum_start_frac = curriculum_start_frac
        self.curriculum_increment_frac = curriculum_increment_frac
        self.curriculum_radius = curriculum_start_frac * self.max_distance
        self.curriculum_increment = curriculum_increment_frac * self.max_distance

        # 5) Éxitos consecutivos para subir nivel
        self.required_consecutive_successes = required_consecutive_successes
        self.success_consecutive = 0
        self.episode_count = 0  # conteo total de episodios exitosos

        # Inicializar estado de episodio
        self._init_episode_state()

        # Inicializar generador de números aleatorios
        self.np_random, _ = seeding.np_random(None)

    def seed(self, seed=None):
        """
        Fija la semilla para reproducibilidad.
        Debe llamarse ANTES de envolver el entorno en Monitor.
        """
        self.np_random, seed_val = seeding.np_random(seed)
        return [seed_val]

    def _init_episode_state(self):
        """Reinicia variables internas al comenzar cada episodio."""
        self.current_step = 0
        self.joint_angles = None
        self.current_pos = None
        self.target_pos = None
        self.prev_distance = None

    def _compute_reward(self, distance, prev_distance, action, hit_limits):
        """
        Reward = 
          (prev_distance – distance) × 30
        + 15 / (1 + 25 × distance²)
        – 0.15 × ‖action‖²
        – 0.4 (si golpea límites)
        + 100 (si distance ≤ goal_threshold)
        """
        improvement = (prev_distance - distance) * 30.0 if prev_distance is not None else 0.0
        proximity = 15.0 / (1.0 + 25.0 * distance**2)
        smoothness = -0.15 * np.sum(np.square(action))
        limits_penalty = -0.4 if hit_limits else 0.0
        success_bonus = 100.0 if distance <= self.goal_threshold else 0.0
        return improvement + proximity + smoothness + limits_penalty + success_bonus

    def _update_curriculum(self):
        """
        Cada vez que se acumulan `required_consecutive_successes` éxitos en fila,
        incrementa `curriculum_radius` en `curriculum_increment` (sin pasarse de max_distance),
        y resetea `success_consecutive` a 0.
        """
        self.episode_count += 1
        self.curriculum_radius = min(
            self.curriculum_radius + self.curriculum_increment,
            self.max_distance
        )
        self.success_consecutive = 0

    def _sample_target(self):
        """
        Muestrea un punto meta tal que su distancia a `self.current_pos` ≤ `curriculum_radius`:
          1) Calcula distancias = ‖workspace_points – current_pos‖
          2) Aplica máscara = (distancias ≤ curriculum_radius)
          3) Si no hay puntos, usa todo workspace_points
          4) Escoge uno al azar y le añade ruido uniforme ±0.02
        """
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
        Verifica si `point` está en el workspace alcanzable:
        - Calcula todas las distancias = ‖workspace_points – point‖
        - Retorna True si min(distancias) ≤ tol, False en otro caso.
        """
        dists = np.linalg.norm(self.workspace_points - point[None, :], axis=1)
        min_dist = np.min(dists)
        return (min_dist <= tol)

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno para un nuevo episodio:
          1) Ángulos aleatorios en [low, high]
          2) Calcula `current_pos` con cinemática directa
          3) Si options['target_pos'], valida con is_reachable() y lo usa; 
             de lo contrario, llama a _sample_target()
          4) `prev_distance` = distancia inicial
          5) Devuelve (observation, info)
        """
        super().reset(seed=seed)
        self._init_episode_state()

        # 1) Ángulos aleatorios
        self.joint_angles = self.np_random.uniform(
            self.joint_limits_low,
            self.joint_limits_high
        ).astype(np.float32)

        # 2) Calcular posición del efector
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
        Ejecuta un paso en el entorno:
          1) Aplica `action` (delta de ángulos) y recorta a [low, high]
          2) Calcula `new_pos` con cinemática directa
          3) `distance` = ‖target_pos – new_pos‖
             `hit_limits` = True si nuevos ángulos chocan con límites
          4) `reward` = _compute_reward(...)
          5) Actualiza `joint_angles`, `current_pos`, `prev_distance`
          6) `terminated` = (distance ≤ goal_threshold)
             `truncated` = (current_step ≥ max_steps)
          7) Si terminated → `success_consecutive += 1`; si alcanza `required_consecutive_successes`, llama a `_update_curriculum()`
             Si no terminated → `success_consecutive = 0`
          8) Construye `observation` e `info` y los retorna junto a `reward`, `terminated`, `truncated`.
        """
        self.current_step += 1

        # 1) Aplicar y recortar ángulos
        new_angles = self.joint_angles + action.astype(np.float32)
        new_angles = np.clip(new_angles, self.joint_limits_low, self.joint_limits_high)

        # 2) Calcular nueva posición del efector
        joint_dict = dict(zip(self.joint_keys, new_angles))
        positions = calculate_joint_positions(
            side=self.side,
            joint_angles=joint_dict
        )
        new_pos = positions[-1].astype(np.float32)

        # 3) Distancia al objetivo y colisión con límites
        distance = np.linalg.norm(self.target_pos - new_pos)
        hit_limits = np.any(
            (new_angles <= self.joint_limits_low + 1e-6) |
            (new_angles >= self.joint_limits_high - 1e-6)
        )

        # 4) Calcular recompensa
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

        # 6) Determinar si termina
        terminated = bool(distance <= self.goal_threshold)
        truncated = bool(self.current_step >= self.max_steps)

        # 7) Gestión de éxitos consecutivos y currículo
        if terminated:
            self.success_consecutive += 1
            if self.success_consecutive >= self.required_consecutive_successes:
                self._update_curriculum()
        else:
            self.success_consecutive = 0

        # 8) Construir observación e `info`
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
            'success_consecutive': int(self.success_consecutive),
            'target_pos': self.target_pos.copy()
        }
        return observation, float(reward), terminated, truncated, info

    def render(self):
        """
        Renderiza en Matplotlib (3D) la pose actual del brazo.
        Solo si `self.render_mode == "human"`.
        """
        if self.render_mode == "human":
            angles_dict = dict(zip(self.joint_keys, self.joint_angles))
            if self.side == 'Left':
                plot_robot(left_angles=angles_dict, right_angles=None)
            else:
                plot_robot(left_angles=None, right_angles=angles_dict)

    def close(self):
        """Placeholder: no hace nada."""
        pass


if __name__ == "__main__":
    """
    Bloque interactivo para probar el entorno desde consola.
    Permite ingresar parámetros vía input() y ejecutar un episodio con acciones aleatorias.
    """
    print("\n=== Prueba interactiva de PepperArmEnv ===")

    # Entrada por consola de parámetros
    side = input("¿Qué brazo entrenar? (Left/Right) [Left]: ").strip() or "Left"
    if side not in ("Left", "Right"):
        print("Entrada inválida, se usará 'Left'.")
        side = "Left"

    try:
        n_samples = int(input("n_workspace_samples (muestras por dimensión) [8]: ") or "8")
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

    print(f"\nCreando entorno con:\n"
          f"  side={side}, n_workspace_samples={n_samples},\n"
          f"  curriculum_start_frac={start_frac}, curriculum_increment_frac={incr_frac},\n"
          f"  max_steps={max_steps}, required_consecutive_successes={req_succ}\n")

    env = PepperArmEnv(
        side=side,
        render_mode="human",
        max_steps=max_steps,
        n_workspace_samples=n_samples,
        curriculum_start_frac=start_frac,
        curriculum_increment_frac=incr_frac,
        required_consecutive_successes=req_succ
    )

    # Ejecutar un episodio con acciones aleatorias
    obs, info = env.reset()
    print(f"Estado inicial:\n"
          f"  Ángulos: {info['joint_angles']}\n"
          f"  Posición inicial: {obs[5:8]}\n"
          f"  Target: {obs[8:11]}\n"
          f"  Distancia inicial: {info['distance']:.4f} m\n"
          f"  Radio curricular: {env.curriculum_radius:.4f} m\n"
          f"  Éxitos consec.: {env.success_consecutive}\n"
         )

    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info_step = env.step(action)
        total_reward += reward
        print(f"Step {env.current_step:03d} → dist={info_step['distance']:.4f} m, "
              f"reward={reward:.2f}, success={info_step['is_success']}, "
              f"consec_success={info_step['success_consecutive']}, "
              f"curric_radius={info_step['curriculum_radius']:.4f}")
        env.render()

        if terminated or truncated:
            done = True

    if info_step['is_success']:
        print(f"\n¡Éxito! Recompensa total={total_reward:.2f}")
    else:
        print(f"\nFracaso (truncado). Distancia final={info_step['distance']:.4f} m")

    print("\nFin de la prueba interactiva.\n")
