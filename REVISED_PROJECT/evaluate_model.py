# evaluate_model.py

import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO, SAC
from pepper_env import PepperArmEnv
from scripts.robot_graph import plot_robot

def evaluate(model_name: str, algorithm: str = "PPO"):
    base = "resultados_calibracion"
    folder = os.path.join(base, model_name)
    model_path = os.path.join(folder, "final_model.zip")

    # 1) Cargar el modelo
    if algorithm.upper() == "PPO":
        model = PPO.load(model_path)
    else:
        model = SAC.load(model_path)

    # 2) Crear entorno idéntico al de entrenamiento
    env = PepperArmEnv(
        side="Left",
        render_mode=None,
        max_steps=250,
        n_workspace_samples=8,
        curriculum_start_frac=0.2,
        curriculum_increment_frac=0.1,
        required_consecutive_successes=5
    )

    # 3) Ejecutar un episodio de prueba
    obs, info = env.reset()
    target_pos = obs[8:11].copy()
    joint_history = [obs[0:5].copy()]
    start_time = time.time()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        joint_history.append(info["joint_angles"])
        done = terminated or truncated

    elapsed = time.time() - start_time
    joint_history = np.array(joint_history)  # (T, 5)
    final_angles = joint_history[-1]
    achieved_pos = env.current_pos.copy()
    final_distance = np.linalg.norm(target_pos - achieved_pos)

    # 4) Imprimir lista de ángulos y posiciones
    print(f"\n=== Resultados de evaluación para {model_name} ({algorithm}) ===\n")
    print("Ángulos finales por articulación:")
    for name, angle in zip(env.joint_keys, final_angles):
        print(f"  {name}: {angle:.4f} rad")
    print(f"\nPosición deseada (target): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
    print(f"Posición alcanzada:     [{achieved_pos[0]:.3f}, {achieved_pos[1]:.3f}, {achieved_pos[2]:.3f}] m")
    print(f"Distancia final al target: {final_distance:.4f} m")
    print(f"Tiempo para alcanzar:      {elapsed:.3f} s\n")

    # 5) Ploteo de ángulos vs timestep con nombres reales
    plt.figure()
    for idx, name in enumerate(env.joint_keys):
        plt.plot(joint_history[:, idx], label=name)
    plt.xlabel("Timestep")
    plt.ylabel("Ángulo de articulación [rad]")
    plt.title(f"{model_name}: Evolución de ángulos de las articulaciones")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # 6) Ploteo 3D: alcanzada vs deseada
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # Robot en pose final
    angles_dict = dict(zip(env.joint_keys, final_angles))
    plot_robot(left_angles=angles_dict, right_angles=None, ax=ax1)
    ax1.set_title("Pose alcanzada")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # Marcamos la posición target
    ax2.scatter(*target_pos, marker="x", s=100, c="red", label="Target")
    # También marcamos la posición alcanzada
    ax2.scatter(*achieved_pos, marker="o", s=50, c="blue", label="Alcanzada")
    ax2.set_title("Comparación de posiciones")
    ax2.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluación de un modelo entrenado (PPO o SAC)"
    )
    p.add_argument(
        "model_name",
        help="Nombre de la carpeta del experimento (e.g. PPO-1)"
    )
    p.add_argument(
        "--alg",
        choices=["PPO", "SAC"],
        default="PPO",
        help="Algoritmo con que fue entrenado el modelo"
    )
    args = p.parse_args()
    evaluate(args.model_name, algorithm=args.alg)
