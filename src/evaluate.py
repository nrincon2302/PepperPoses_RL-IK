"""
Evaluación interactiva de modelos entrenados para la cinemática inversa del robot Pepper.
La gráfica se actualiza en la misma ventana sin cerrarla, y se muestran mensajes en consola
para indicar qué está sucediendo en cada paso.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # ◆ Usaremos plt.ion() para modo interactivo

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from pepper_env import PepperArmEnv  # ◆ Aquí viene el render “por defecto” que crea nuevas figuras cada vez :contentReference[oaicite:0]{index=0}
from scripts.robot_graph import plot_robot  # ◆ Función para dibujar el robot en 3D

# Configuración de evaluación
CONFIG = {
    'model_dir': 'models',
    'eval_episodes': 20,
    'render': True,
    'sleep_time': 0.05,        # segundos entre pasos
    'save_results': True,
    'test_specific_targets': True  # Probar poses específicas además de aleatorias
}


def make_env(side='Left', render_mode='human'):
    """Crea un entorno PepperArmEnv configurado para evaluación."""
    def _init():
        env = PepperArmEnv(
            side=side,
            render_mode=render_mode,
            max_steps=250
        )
        return Monitor(env)
    return _init


def get_unwrapped_env(env):
    """Helper para obtener el entorno base sin wrappers."""
    if hasattr(env, 'env'):
        return get_unwrapped_env(env.env)
    return env


def evaluate_model(model_path, env, specific_targets=None):
    """
    Evalúa un modelo en el entorno dado.

    Args:
        model_path: Ruta al modelo guardado (ZIP de SB3)
        env: DummyVecEnv que envuelve PepperArmEnv
        specific_targets: Lista de poses objetivo específicas a evaluar (opcional)
    """
    # 1) Cargar modelo PPO
    print(f"\n[CARGANDO MODELO] {os.path.basename(model_path)}")
    model = PPO.load(model_path, env=env)
    # Obtener el entorno base (sin wrappers)
    base_env = get_unwrapped_env(env.envs[0])
    print("[MODELO CARGADO] Comenzando evaluación...\n")

    # 2) Preparar figura de Matplotlib en modo interactivo
    if CONFIG['render']:
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.show()

    # 3) Preparar almacenamiento de resultados
    results = {
        'episode': [], 'reward': [], 'success': [],
        'final_distance': [], 'steps': [], 'target_pos': []
    }

    n_episodes_random = CONFIG['eval_episodes']
    n_total_episodes = n_episodes_random + (len(specific_targets) if specific_targets else 0)

    # 4) Rutina para evaluación en metas aleatorias
    for ep in range(n_episodes_random):
        print(f"[EPISODIO {ep + 1}/{n_total_episodes}]")
        obs = env.reset()  # Desempaquetar correctamente
        obs = obs[0]  # Obtener el primer elemento ya que es VecEnv
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # 4.1) Predecir acción
            action, _ = model.predict(obs, deterministic=True)

            # 4.2) Tomar paso
            step_result = env.step(action)
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                obs = obs[0]  # Obtener el primer elemento
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                obs = obs[0]
            
            total_reward += float(reward)
            steps += 1

            # 4.3) Renderizar en la misma figura (si corresponde)
            if CONFIG['render']:
                ax.cla()
                # Asegurarnos de que tenemos los ángulos correctos
                if hasattr(base_env, 'joint_angles'):
                    current_angles = base_env.joint_angles.tolist()
                else:
                    # Si no podemos obtener los ángulos directamente, los extraemos de obs
                    # Asumiendo que obs tiene la estructura [ángulos, pos_actual, target]
                    current_angles = obs[:base_env.num_joints].tolist()
                
                # Construir diccionario de ángulos
                angle_keys = list(base_env.joint_limits.keys())
                angles_dict = dict(zip(angle_keys, current_angles))
                
                # Dibujar robot
                if base_env.side == 'Left':
                    plot_robot(left_angles=angles_dict, right_angles=None, ax=ax)
                else:
                    plot_robot(left_angles=None, right_angles=angles_dict, ax=ax)
                
                # Mostrar target
                if hasattr(base_env, 'target_pos'):
                    target = base_env.target_pos
                else:
                    # Si no podemos obtener el target directamente, lo extraemos de obs
                    target = obs[-3:]
                
                ax.scatter(*target, c='green', marker='X', s=100, label='Target')
                
                # Configuración del plot
                ax.set_title(f"PepperArm - Episodio {ep+1}/{n_total_episodes}, Paso {steps}")
                ax.set_xlabel('X [m]')
                ax.set_ylabel('Y [m]')
                ax.set_zlabel('Z [m]')
                ax.legend(loc='upper left')
                
                plt.draw()
                plt.pause(0.001)
            
            # 4.4) Mensajes en consola cada 50 pasos
            # 4.5) Pequeña demora si render
            if CONFIG['render']:
                time.sleep(CONFIG['sleep_time'])

        # Al terminar el episodio (done=True)
        info_final = info if isinstance(info, dict) else info[0]
        final_distance = info_final.get('distance', np.inf)
        is_success = info_final.get('is_success', False)

        # 4.6) Guardar resultados
        results['episode'].append(ep)
        results['reward'].append(total_reward)
        results['success'].append(is_success)
        results['final_distance'].append(final_distance)
        results['steps'].append(steps)

        print(f"[EP {ep + 1}] Reward: {total_reward:.2f} | "
              f"Distancia final: {info.get('distance', np.inf):.4f}m | "
              f"Pasos: {steps}")

    # 5) Evaluar en poses específicas si se proporcionaron
    if specific_targets is not None:
        for i, target in enumerate(specific_targets):
            ep_idx = n_episodes_random + i
            print(f"[POSE ESPECÍFICA {i + 1}/{len(specific_targets)}] → Meta fija {target.tolist()}")
            # Reset con meta explícita
            obs = env.reset(options={'target_pos': target})
            done = False
            total_reward = 0.0
            steps = 0

            # En este caso, target es conocido (lo pasamos), así que lo usamos directamente
            target_from_obs = target.copy()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result

                total_reward += float(reward)
                steps += 1

                # Render en la misma figura
                if CONFIG['render']:
                    ax.cla()
                    current_angles = obs[0:5].tolist()
                    # Usar el entorno base para obtener los límites
                    angle_keys = list(base_env.joint_limits.keys())
                    angles_dict = {angle_keys[i]: current_angles[i] for i in range(5)}
                    if env.envs[0].side == 'Left':
                        plot_robot(left_angles=angles_dict, right_angles=None, ax=ax)
                    else:
                        plot_robot(left_angles=None, right_angles=angles_dict, ax=ax)

                    ax.scatter(
                        [target_from_obs[0]], [target_from_obs[1]], [target_from_obs[2]],
                        c='green', marker='X', s=100, label='Target'
                    )
                    ax.set_title(f"PepperArm - Meta Específica {i+1}/{len(specific_targets)}, Paso {steps}")
                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.set_zlabel('Z [m]')
                    ax.legend(loc='upper left')

                    plt.draw()
                    plt.pause(0.001)

                if steps % 50 == 0:
                    curr_dist = info.get('distance', np.nan)
                    print(f"    [POSE {i+1} / STEP {steps}] reward acumulado: {total_reward:.2f} | dist actual: {curr_dist:.4f}m")

                if CONFIG['render']:
                    time.sleep(CONFIG['sleep_time'])

            info_final = info if isinstance(info, dict) else info[0]
            final_distance = info_final.get('distance', np.inf)
            is_success = info_final.get('is_success', False)

            results['episode'].append(ep_idx)
            results['reward'].append(total_reward)
            results['success'].append(is_success)
            results['final_distance'].append(final_distance)
            results['steps'].append(steps)
            results['target_pos'].append(target_from_obs.tolist())

            print(f"[POSE {i + 1} COMPLETADA] Reward: {total_reward:.2f} | Éxito: {is_success} | "
                  f"Distancia final: {final_distance:.4f}m | Pasos: {steps}\n")

    # 6) Convertir a DataFrame y retornar
    return pd.DataFrame(results)


def main():
    # 1) Crear entorno vectorizado
    print("[INICIANDO ENTORNO]")
    env = DummyVecEnv([make_env(render_mode="human" if CONFIG['render'] else None)])
    print("[ENTORNO LISTO]\n")

    # 2) Definir poses específicas
    specific_targets = [
        np.array([0.2, 0.2, 0.8]),
        np.array([0.2, -0.2, 0.8]),
        np.array([0.3, 0.0, 0.6])
    ] if CONFIG['test_specific_targets'] else None

    # 3) Encontrar modelo más reciente en 'models/ppo_pepper_arm.zip'
    latest_model = os.path.join(CONFIG['model_dir'], 'ppo_pepper_arm.zip')
    if not os.path.exists(latest_model):
        raise FileNotFoundError(f"No se encontró el modelo en: {latest_model}")
    print(f"[MODELO DETECTADO] {latest_model}\n")

    # 4) Evaluar y recolectar resultados
    results_df = evaluate_model(latest_model, env, specific_targets)

    # 5) Guardar resultados en CSV si corresponde
    if CONFIG['save_results']:
        results_path = os.path.join(CONFIG['model_dir'], 'evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"[RESULTADOS GUARDADOS] {results_path}\n")

    # 6) Estadísticas finales
    print("[ESTADÍSTICAS FINALES DE EVALUACIÓN]")
    print("-" * 50)
    print(f"Tasa de éxito: {results_df['success'].mean() * 100:.1f}%")
    print(f"Distancia final media: {results_df['final_distance'].mean():.4f}m ± "
          f"{results_df['final_distance'].std():.4f}m")
    print(f"Pasos promedio: {results_df['steps'].mean():.1f} ± {results_df['steps'].std():.1f}")
    print("\n[EVALUACIÓN COMPLETADA]")

    env.close()
    # Si usaste plt.ion(), conviene apagar el modo interactivo
    if CONFIG['render']:
        plt.ioff()
        plt.show()  # Para mantener la última figura abierta


if __name__ == "__main__":
    main()
