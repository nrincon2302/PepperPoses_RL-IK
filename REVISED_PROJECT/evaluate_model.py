import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO, SAC
from pepper_env import PepperArmEnv
from scripts.CSpace import generate_workspace_points
from scripts.robot_graph import plot_robot

def obtener_targets(base_dir, n_tests, semilla=0):
    np.random.seed(semilla)
    ruta = os.path.join(base_dir, "test_targets.npy")
    if os.path.exists(ruta):
        return np.load(ruta)
    else:
        espacio, _ = generate_workspace_points(side="Left", n_samples=8)
        idxs = np.random.choice(len(espacio), size=n_tests, replace=False)
        targets = espacio[idxs]
        np.save(ruta, targets)
        return targets

def evaluar(modelo, targets):
    base = "resultados_calibracion"
    carpeta = os.path.join(base, modelo)
    ruta_modelo = os.path.join(carpeta, "final_model.zip")
    alg = modelo.split("-")[0]

    # Cargar modelo
    agente = PPO.load(ruta_modelo) if alg == "PPO" else SAC.load(ruta_modelo)

    # Entorno
    env = PepperArmEnv(
        side="Left", render_mode=None,
        max_steps=250, n_workspace_samples=8,
        curriculum_start_frac=0.2, curriculum_increment_frac=0.2,
        required_consecutive_successes=2
    )

    errores = []
    tiempos = []
    exitos = 0
    info_ultima = None

    for tgt in targets:
        obs0, _ = env.reset(options={"target_pos": tgt})
        # Guardamos el estado inicial real antes de cualquier acción
        ang_ini = obs0[0:5].copy()
        t0 = time.time()
        terminado = False

        while not terminado:
            accion, _ = agente.predict(obs0, deterministic=True)
            obs0, _, term, trunc, info = env.step(accion)
            terminado = term or trunc

        dur = time.time() - t0
        err = info["distance"]
        tiempos.append(dur)
        errores.append(err)
        if err <= 0.10:
            exitos += 1

        info_ultima = (tgt.copy(), env.current_pos.copy(),
                       ang_ini, info["joint_angles"].copy(),
                       dur, err)

    # Estadísticas
    err_arr = np.array(errores)
    time_arr = np.array(tiempos)
    print(f"\n=== Evaluación de {modelo} ({alg}) ===")
    print(f"Pruebas totales     : {len(targets)}")
    print(f"Éxitos (≤0.10 m)    : {exitos} ({exitos/len(targets)*100:.1f} %)")
    print(f"Error promedio      : {err_arr.mean():.4f} m ± {err_arr.std():.4f}")
    print(f"Tiempo promedio     : {time_arr.mean():.3f} s ± {time_arr.std():.3f}\n")


    # Evolución de ángulos (última prueba)
    obs, _ = env.reset(options={"target_pos": tgt})
    historial = [obs[0:5].copy()]
    term = False
    while not term:
        accion, _ = agente.predict(obs, deterministic=True)
        obs, _, tr, trun, _ = env.step(accion)
        historial.append(obs[0:5].copy())
        term = tr or trun
    historial = np.array(historial)

    # Detalle última prueba
    tgt, ach, ang_ini, ang_fin, dur, err = info_ultima
    print("Última prueba:")
    print("  Ángulos iniciales → finales:")
    for nombre, ai, af in zip(env.joint_keys, historial[0], ang_fin):
        print(f"    {nombre}: {ai:.4f} → {af:.4f} rad")
    print(f"  Posición deseada : [{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]")
    print(f"  Posición alcanzada: [{ach[0]:.3f}, {ach[1]:.3f}, {ach[2]:.3f}]")
    print(f"  Error final      : {err:.4f} m")
    print(f"  Tiempo           : {dur:.3f} s\n")

    plt.figure()
    for i, nombre in enumerate(env.joint_keys):
        plt.plot(historial[:, i], label=nombre)
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Ángulo [rad]")
    plt.title(f"{modelo}: Evolución de ángulos (última prueba)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Comparativo 3D (última prueba)
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121, projection="3d")
    plot_robot(left_angles=dict(zip(env.joint_keys, historial[-1])), right_angles=None, ax=ax1)
    ax1.set_title("Pose alcanzada")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(*tgt, marker="x", s=100, c="red", label="Deseada")
    ax2.scatter(*ach, marker="o", s=50, c="blue", label="Alcanzada")
    ax2.set_title("Meta vs Alcanzada")
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar múltiples modelos con mismo banco de pruebas")
    parser.add_argument("--models", nargs="+", default=['PPO-1', 'PPO-2', 'PPO-3', 'PPO-4', 'PPO-5', 'SAC-1', 'SAC-2', 'SAC-3'],
                        help="Lista de carpetas (ej.: PPO-1 SAC-2)")
    parser.add_argument("--n_tests", type=int, default=20,
                        help="Cantidad de pruebas en el banco")
    parser.add_argument("--seed", type=int, default=0,
                        help="Semilla para generar banco de pruebas")
    args = parser.parse_args()

    banco = obtener_targets("resultados_calibracion", args.n_tests, semilla=args.seed)
    for modelo in args.models:
        evaluar(modelo, banco)
