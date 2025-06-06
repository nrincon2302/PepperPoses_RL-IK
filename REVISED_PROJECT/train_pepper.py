# File: train_pepper.py

import os
import argparse
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure

from pepper_env import PepperArmEnv

RESULTS_DIR = "resultados_calibracion"


class CurriculumLoggingCallback(BaseCallback):
    """
    Callback custom para registrar métricas específicas al final de cada episodio:
      - episodio, recompensa total, longitud, éxito/fracaso,
      - curriculum_radius al final, éxitos consecutivos,
      - última posición del efector y ángulos de las articulaciones,
      - target actual.
    Guarda una fila en un CSV dentro de la carpeta del experimento.
    """

    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        # Si el CSV no existe, lo creamos con encabezados
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                "episode",
                "total_reward",
                "episode_length",
                "is_success",
                "curriculum_radius",
                "consecutive_successes",
                "final_joint_angles",
                "target_pos"
            ])
            df.to_csv(self.csv_path, index=False)

    def _on_step(self) -> bool:
        # Se ejecuta cada vez que Monitor detecta fin de episodio (info["episode"]).
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        for info in infos:
            if "episode" in info.keys():
                epi_data = info["episode"]
                total_reward = epi_data["r"]
                length = epi_data["l"]
                is_success = info.get("is_success", False)
                curr_rad = info.get("curriculum_radius", None)
                consec_succ = info.get("success_consecutive", None)
                final_angles = info.get("joint_angles", None)
                target_pos = info.get("target_pos", None)

                final_angles_str = (
                    np.array2string(np.array(final_angles), separator=",")
                    if final_angles is not None else ""
                )
                target_pos_str = (
                    np.array2string(np.array(target_pos), separator=",")
                    if target_pos is not None else ""
                )

                df_all = pd.read_csv(self.csv_path)
                next_ep = len(df_all) + 1

                new_row = {
                    "episode": next_ep,
                    "total_reward": total_reward,
                    "episode_length": length,
                    "is_success": int(is_success),
                    "curriculum_radius": float(curr_rad) if curr_rad is not None else np.nan,
                    "consecutive_successes": int(consec_succ) if consec_succ is not None else np.nan,
                    "final_joint_angles": final_angles_str,
                    "target_pos": target_pos_str
                }
                df_all = df_all.append(new_row, ignore_index=True)
                df_all.to_csv(self.csv_path, index=False)
        return True


def make_env(seed: int, log_folder: str, **env_kwargs):
    """
    Crea una instancia de PepperArmEnv, le fija la semilla al entorno base,
    y luego lo envuelve en Monitor para logging.

    Args:
        - seed (int): semilla para reproducibilidad.
        - log_folder (str): carpeta donde Monitor guardará 'monitor.csv'.
        - env_kwargs: argumentos para PepperArmEnv (side, max_steps, etc.).

    Devuelve:
        - Un entorno tipo Monitor(PepperArmEnv).
    """
    # 1) Crear el entorno base y fijar semilla
    env_raw = PepperArmEnv(**env_kwargs)
    env_raw.seed(seed)

    # 2) Envolverlo en Monitor para que genere 'monitor.csv'
    monitor_path = os.path.join(log_folder, "monitor.csv")
    os.makedirs(log_folder, exist_ok=True)
    env = Monitor(env_raw, filename=monitor_path)
    return env


def optimize_ppo(trial: optuna.trial.Trial, env_kwargs, total_timesteps: int, result_folder: str):
    """
    Función objetivo para Optuna (PPO).
    Trial escoge hiperparámetros, entrenamos PPO con estos y evaluamos recompensa media
    en un conjunto de episodios de validación. Guardamos modelo y logs en result_folder.
    """
    # Espacio de búsqueda PPO
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    gamma = trial.suggest_uniform("gamma", 0.95, 0.9999)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-2)
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
    vf_coef = trial.suggest_uniform("vf_coef", 0.1, 1.0)

    # Crear carpeta para este trial
    trial_id = trial.number + 1
    alg_folder = os.path.join(result_folder, f"PPO-{trial_id}")
    os.makedirs(alg_folder, exist_ok=True)

    # Logger de SB3 (stdout, CSV y TensorBoard)
    new_logger = configure(os.path.join(alg_folder, "tb_logs"), ["stdout", "csv", "tensorboard"])

    # Entornos de entrenamiento y evaluación
    train_env = make_env(seed=trial_id, log_folder=alg_folder, **env_kwargs)
    eval_env = make_env(seed=trial_id + 1000, log_folder=alg_folder, **env_kwargs)

    # Callbacks
    csv_path = os.path.join(alg_folder, "curriculum_metrics.csv")
    curriculum_cb = CurriculumLoggingCallback(csv_path=csv_path)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=alg_folder,
        log_path=os.path.join(alg_folder, "eval_logs"),
        eval_freq=total_timesteps // 10,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0
    )

    # Crear modelo PPO con los hiperparámetros del trial
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=64,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        vf_coef=vf_coef,
        verbose=0,
        tensorboard_log=os.path.join(alg_folder, "tb_logs")
    )
    model.set_logger(new_logger)

    # Barra de progreso
    total_iter = total_timesteps // n_steps
    with tqdm(total=total_timesteps, desc=f"PPO Trial {trial_id}", unit="step") as pbar:
        class TqdmStepCallback(BaseCallback):
            def __init__(self, pbar, verbose=0):
                super().__init__(verbose)
                self.pbar = pbar
                self.n_steps = n_steps

            def _on_step(self):
                # Actualizamos pbar en incrementos de n_steps
                self.pbar.update(self.n_steps)
                return True

        tqdm_cb = TqdmStepCallback(pbar)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[tqdm_cb, curriculum_cb, eval_callback]
        )

    # Guardar modelo final
    model.save(os.path.join(alg_folder, "final_model"))

    # Evaluar rendimiento final (media de recompensa)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    return mean_reward


def optimize_sac(trial: optuna.trial.Trial, env_kwargs, total_timesteps: int, result_folder: str):
    """
    Función objetivo para Optuna (SAC).
    """
    # Espacio de búsqueda SAC
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_categorical("buffer_size", [100_000, 300_000, 500_000])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    tau = trial.suggest_uniform("tau", 0.005, 0.05)
    gamma = trial.suggest_uniform("gamma", 0.95, 0.9999)
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8])
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-2)

    # Crear carpeta para este trial
    trial_id = trial.number + 1
    alg_folder = os.path.join(result_folder, f"SAC-{trial_id}")
    os.makedirs(alg_folder, exist_ok=True)

    # Logger
    new_logger = configure(os.path.join(alg_folder, "tb_logs"), ["stdout", "csv", "tensorboard"])

    # Entornos
    train_env = make_env(seed=trial_id, log_folder=alg_folder, **env_kwargs)
    eval_env = make_env(seed=trial_id + 1000, log_folder=alg_folder, **env_kwargs)

    csv_path = os.path.join(alg_folder, "curriculum_metrics.csv")
    curriculum_cb = CurriculumLoggingCallback(csv_path=csv_path)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=alg_folder,
        log_path=os.path.join(alg_folder, "eval_logs"),
        eval_freq=total_timesteps // 10,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        ent_coef=ent_coef,
        verbose=0,
        tensorboard_log=os.path.join(alg_folder, "tb_logs")
    )
    model.set_logger(new_logger)

    # Barra de progreso
    with tqdm(total=total_timesteps, desc=f"SAC Trial {trial_id}", unit="step") as pbar:
        class TqdmStepCallback(BaseCallback):
            def __init__(self, pbar, verbose=0):
                super().__init__(verbose)
                self.pbar = pbar

            def _on_step(self):
                # Actualizamos la barra en incrementos de 1 timestep
                self.pbar.update(1)
                return True

        tqdm_cb = TqdmStepCallback(pbar)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[tqdm_cb, curriculum_cb, eval_callback]
        )

    # Guardar modelo final
    model.save(os.path.join(alg_folder, "final_model"))

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    return mean_reward


def run_hpo(alg_name: str, env_kwargs: dict, total_timesteps: int, n_trials: int):
    """
    Ejecuta búsqueda de hiperparámetros (Optuna) para el algoritmo elegido.
    alg_name ∈ {'PPO', 'SAC'}.
    Guarda resultados en carpetas bajo RESULTS_DIR.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    study = optuna.create_study(
        sampler=TPESampler(),
        direction="maximize",
        study_name=f"{alg_name}_Study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    if alg_name == "PPO":
        objective = lambda trial: optimize_ppo(trial, env_kwargs, total_timesteps, RESULTS_DIR)
    else:
        objective = lambda trial: optimize_sac(trial, env_kwargs, total_timesteps, RESULTS_DIR)

    study.optimize(objective, n_trials=n_trials)

    study.trials_dataframe().to_csv(os.path.join(RESULTS_DIR, f"{alg_name}_hpo_results.csv"))
    print(f"\n=== HPO {alg_name} completado: mejores parámetros ===")
    print(study.best_params)
    print(f"Valor objetivo (recompensa media): {study.best_value:.2f}\n")


def evaluate_policy(model, env, n_eval_episodes: int = 5):
    """
    Evalúa la política en env durante n_eval_episodes y retorna (mean_reward, std_reward).
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
        episode_rewards.append(total_r)
    return np.mean(episode_rewards), np.std(episode_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento HPO con SB3 para PepperArmEnv")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Número total de timesteps por trial (p.ej. 500000).")
    parser.add_argument("--trials", type=int, default=10,
                        help="Número de trials de HPO por algoritmo.")
    parser.add_argument("--side", type=str, default="Left", choices=["Left", "Right"],
                        help="Brazo a entrenar: 'Left' o 'Right'.")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="n_workspace_samples (p. ej. 8).")
    parser.add_argument("--start_frac", type=float, default=0.2,
                        help="Fracción inicial del currículo (p.ej. 0.2).")
    parser.add_argument("--incr_frac", type=float, default=0.1,
                        help="Fracción de incremento del currículo (p.ej. 0.1).")
    parser.add_argument("--max_steps", type=int, default=250,
                        help="Máximo de pasos por episodio.")
    parser.add_argument("--required_succ", type=int, default=5,
                        help="Éxitos consecutivos necesarios para subir nivel.")
    args = parser.parse_args()

    env_kwargs = {
        "side": args.side,
        "render_mode": None,
        "max_steps": args.max_steps,
        "n_workspace_samples": args.n_samples,
        "curriculum_start_frac": args.start_frac,
        "curriculum_increment_frac": args.incr_frac,
        "required_consecutive_successes": args.required_succ
    }

    print("\n=== Iniciando búsqueda de hiperparámetros ===")
    print(f"Configuración común: {env_kwargs}")
    print(f"Timesteps por trial: {args.timesteps}, Trials: {args.trials}\n")

    # 1) HPO para PPO
    run_hpo("PPO", env_kwargs, args.timesteps, args.trials)

    # 2) HPO para SAC
    run_hpo("SAC", env_kwargs, args.timesteps, args.trials)

    print("\n=== Entrenamiento y HPO completados ===\n")
