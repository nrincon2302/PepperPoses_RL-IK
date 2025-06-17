import os
import time
import argparse
import numpy as np
import pybullet as p
from stable_baselines3 import PPO, SAC
from pepper_env import PepperArmEnv
import matplotlib.pyplot as plt

def obtener_targets(base_dir, n_tests, side, semilla=42):
    # Cargar targets desde el caché del workspace para consistencia
    cache_path = os.path.join("workspace_cache", f"workspace_{side.lower()}_8.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError("Ejecuta 'generate_workspace_cache.py' primero.")
    
    workspace_points = np.load(cache_path)
    np.random.seed(semilla)
    idxs = np.random.choice(len(workspace_points), size=n_tests, replace=False)
    return workspace_points[idxs]

def evaluar(modelo_folder: str, targets: np.ndarray, side: str, show_gui: bool):
    base = "resultados_calibracion"
    carpeta = os.path.join(base, modelo_folder)
    ruta_modelo = os.path.join(carpeta, "best_model.zip")
    if not os.path.exists(ruta_modelo):
        ruta_modelo = os.path.join(carpeta, "final_model.zip")
        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(f"No se encontró el modelo en {carpeta}")
    
    alg = modelo_folder.split("-")[0].upper()
    agente = PPO.load(ruta_modelo) if alg == "PPO" else SAC.load(ruta_modelo)
    env = PepperArmEnv(side=side, render_mode="human" if show_gui else None)
    
    all_results = []
    print(f"\n=== Evaluación de {modelo_folder} ({alg}) ===")
    for i, tgt in enumerate(targets):
        if show_gui:
            print(f"Prueba {i+1}/{len(targets)}. Target: {tgt.round(3)}. Presiona Enter...")
            input()

        obs, info = env.reset(options={"target_pos": tgt})
        angles_history = [info['joint_angles'].copy()]
        done = False
        
        while not done:
            action, _ = agente.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            angles_history.append(info['joint_angles'].copy())
            done = terminated or truncated
            if show_gui: time.sleep(1.0 / env.control_frequency)
        
        all_results.append({'distance': info["distance"], 'is_success': info['is_success'], 'angles_history': np.array(angles_history)})
        print(f"  Resultado: Error final={info['distance']:.4f} m, Éxito={info['is_success']}")

    env.close()

    # Estadísticas
    success_rate = np.mean([res['is_success'] for res in all_results]) * 100
    avg_error = np.mean([res['distance'] for res in all_results])
    print(f"\n--- Resumen de {modelo_folder} ---")
    print(f"Tasa de éxito ({env.goal_threshold*100:.1f} cm): {success_rate:.1f} %")
    print(f"Error promedio      : {avg_error:.4f} m")

    # Gráfica de la última prueba
    last_history = all_results[-1]['angles_history']
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(env.joint_keys):
        plt.plot(last_history[:, i], label=name)
    
    plt.title(f"Evolución de Ángulos - {modelo_folder} (Última Prueba)")
    plt.xlabel("Paso de tiempo"); plt.ylabel("Ángulo [rad]")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="Carpetas de modelos (ej: SAC-0)")
    parser.add_argument("--side", choices=["Left", "Right"], default="Left")
    parser.add_argument("--n_tests", type=int, default=20)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    banco_pruebas = obtener_targets("resultados_calibracion", args.n_tests, args.side)
    for modelo in args.models:
        evaluar(modelo, banco_pruebas, args.side, not args.no_gui)