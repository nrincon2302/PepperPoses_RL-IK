import os
import time
import argparse
import numpy as np
import pybullet as p

from stable_baselines3 import PPO, SAC
from environments.pepper_env import PepperArmEnv
from scripts.CSpace import generate_workspace_points


def obtener_targets(base_dir, n_tests, side, semilla=0):
    np.random.seed(semilla)
    ruta = os.path.join(base_dir, f"test_targets_{side}.npy")
    if os.path.exists(ruta):
        return np.load(ruta)
    else:
        # Asegurarse de que el directorio existe
        os.makedirs(base_dir, exist_ok=True)
        espacio, _ = generate_workspace_points(side=side, n_samples=8)
        idxs = np.random.choice(len(espacio), size=n_tests, replace=False)
        targets = espacio[idxs]
        np.save(ruta, targets)
        return targets

def evaluar(modelo_folder: str, targets: np.ndarray, side: str, show_gui: bool):
    base = f"resultados_{side}"
    carpeta = os.path.join(base, modelo_folder)
    ruta_modelo = os.path.join(carpeta, "best_model.zip")
    if not os.path.exists(ruta_modelo):
        ruta_modelo = os.path.join(carpeta, "final_model.zip")
    
    alg = modelo_folder.split("-")[0].upper()
    agente = PPO.load(ruta_modelo) if alg == "PPO" else SAC.load(ruta_modelo)

    # Crear entorno
    env = PepperArmEnv(side=side, render_mode="human" if show_gui else None)
    
    target_vis_id = -1
    errores, tiempos, exitos = [], [], 0
    
    print(f"\n=== Evaluación de {modelo_folder} ({alg}) ===")
    for i, tgt in enumerate(targets):
        if show_gui and target_vis_id != -1:
            p.removeBody(target_vis_id, physicsClientId=env.client)

        if show_gui:
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.025,
                rgbaColor=[1, 0, 0, 0.8],
                physicsClientId=env.client
            )
            target_vis_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=tgt.tolist(),
                physicsClientId=env.client
            )
            
            print(f"Prueba {i+1}/{len(targets)}. Target: {tgt}. Presiona Enter para continuar...")
            input()

        obs, _ = env.reset(options={"target_pos": tgt})
        done, total_steps = False, 0
        t0 = time.time()
        
        while not done:
            action, _ = agente.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
            if show_gui:
                time.sleep(1/30)

        err = info["distance"]
        errores.append(err)
        tiempos.append(time.time() - t0)
        if info['is_success']:
            exitos += 1
        
        print(f"  Resultado: Error final={err:.4f} m, Pasos={total_steps}, Éxito={info['is_success']}")

    env.close()

    # Estadísticas
    err_arr = np.array(errores)
    time_arr = np.array(tiempos)
    print(f"\n--- Resumen de {modelo_folder} ---")
    print(f"Pruebas totales     : {len(targets)}")
    print(f"Éxitos (≤{env.goal_threshold:.2f} m)    : {exitos} ({exitos/len(targets)*100:.1f} %)")
    print(f"Error promedio      : {err_arr.mean():.4f} m ± {err_arr.std():.4f}")
    print(f"Tiempo promedio/ep  : {time_arr.mean():.3f} s ± {time_arr.std():.3f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelos entrenados")
    parser.add_argument("--models", nargs="+", required=True, help="Lista de carpetas de modelos (ej: SAC-0 PPO-1)")
    parser.add_argument("--side", choices=["Left", "Right"], default="Left", help="Brazo a evaluar")
    parser.add_argument("--n_tests", type=int, default=1000, help="Cantidad de targets a probar")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para generar banco de pruebas")
    parser.add_argument("--no-gui", action="store_true", help="Ejecutar evaluación sin GUI")
    args = parser.parse_args()

    banco_pruebas = obtener_targets("resultados_calibracion", args.n_tests, args.side, semilla=args.seed)
    
    for modelo in args.models:
        evaluar(modelo, banco_pruebas, args.side, not args.no_gui)