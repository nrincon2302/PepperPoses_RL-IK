import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure

from pepper_env import PepperArmEnv

RESULTS_DIR = "resultados_calibracion"


class CurriculumLoggingCallback(BaseCallback):
    """
    Callback para registrar métricas de currículo al final de cada episodio.
    Genera un CSV con columnas:
      - episode, total_reward, episode_length, is_success,
      - curriculum_radius, consecutive_successes,
      - final_joint_angles, target_pos
    """

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        # Si no existe, creamos el CSV con cabeceras
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
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # Leemos el CSV existente
        df_all = pd.read_csv(self.csv_path)

        for info in infos:
            if "episode" in info:
                epi = info["episode"]
                total_reward = epi["r"]
                length = epi["l"]
                is_success = info.get("is_success", False)
                curr_rad = info.get("curriculum_radius", np.nan)
                consec = info.get("success_consecutive", np.nan)
                angles = info.get("joint_angles", [])
                target = info.get("target_pos", [])

                angles_str = np.array2string(np.array(angles), separator=",")
                target_str = np.array2string(np.array(target), separator=",")

                new_row = {
                    "episode": len(df_all) + 1,
                    "total_reward": total_reward,
                    "episode_length": length,
                    "is_success": int(is_success),
                    "curriculum_radius": float(curr_rad),
                    "consecutive_successes": int(consec),
                    "final_joint_angles": angles_str,
                    "target_pos": target_str
                }
                df_all = pd.concat([df_all, pd.DataFrame([new_row])], ignore_index=True)

        df_all.to_csv(self.csv_path, index=False)
        return True


def make_env(seed: int, log_folder: str, **env_kwargs):
    """
    Crea un PepperArmEnv y lo envuelve en Monitor.
    Fija la semilla ANTES de envolverlo.
    """
    os.makedirs(log_folder, exist_ok=True)
    env_raw = PepperArmEnv(**env_kwargs)
    env_raw.seed(seed)
    monitor_path = os.path.join(log_folder, "monitor.csv")
    return Monitor(env_raw, filename=monitor_path)


def evaluate_policy(model, env, n_eval_episodes: int = 5):
    """
    Ejecuta la política determinística durante n_eval_episodes y devuelve (mean_reward, std_reward).
    """
    rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            done = terminated or truncated
        rewards.append(total_r)
    return np.mean(rewards), np.std(rewards)


def optimize_ppo(trial: optuna.Trial, env_kwargs, total_timesteps: int, base_dir: str):
    """
    Función objetivo PPO para Optuna.
    """
    # Espacio de búsqueda
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    gamma = trial.suggest_uniform("gamma", 0.95, 0.9999)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-2)
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
    vf_coef = trial.suggest_uniform("vf_coef", 0.1, 1.0)

    # Carpeta de este trial
    trial_id = trial.number + 1
    alg_folder = os.path.join(base_dir, f"PPO-{trial_id}")
    os.makedirs(alg_folder, exist_ok=True)

    # Logger de SB3
    logger = configure(os.path.join(alg_folder, "tb_logs"), ["stdout", "csv", "tensorboard"])

    # Entorno de entrenamiento y evaluación
    train_env = make_env(seed=trial_id, log_folder=alg_folder, **env_kwargs)
    eval_env = make_env(seed=trial_id + 1000, log_folder=os.path.join(alg_folder, "eval"), **env_kwargs)

    # Callbacks
    curriculum_cb = CurriculumLoggingCallback(csv_path=os.path.join(alg_folder, "curriculum_metrics.csv"))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=alg_folder,
        log_path=os.path.join(alg_folder, "eval_logs"),
        eval_freq=max(1, total_timesteps // 10),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0
    )

    # Crear modelo PPO
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=lr,
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
    model.set_logger(logger)

    # Entrenar con barra de progreso integrada
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([curriculum_cb, eval_cb]),
        progress_bar=True
    )

    # Guardar modelo final
    model.save(os.path.join(alg_folder, "final_model"))

    # Evaluar recompensa media final
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    return mean_reward


def optimize_sac(trial: optuna.Trial, env_kwargs, total_timesteps: int, base_dir: str):
    """
    Función objetivo SAC para Optuna.
    """
    # Espacio de búsqueda
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_categorical("buffer_size", [100_000, 300_000, 500_000])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    tau = trial.suggest_uniform("tau", 0.005, 0.05)
    gamma = trial.suggest_uniform("gamma", 0.95, 0.9999)
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8])
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-2)

    # Carpeta de este trial
    trial_id = trial.number + 1
    alg_folder = os.path.join(base_dir, f"SAC-{trial_id}")
    os.makedirs(alg_folder, exist_ok=True)

    # Logger
    logger = configure(os.path.join(alg_folder, "tb_logs"), ["stdout", "csv", "tensorboard"])

    # Entornos
    train_env = make_env(seed=trial_id, log_folder=alg_folder, **env_kwargs)
    eval_env = make_env(seed=trial_id + 1000, log_folder=os.path.join(alg_folder, "eval"), **env_kwargs)

    # Callbacks
    curriculum_cb = CurriculumLoggingCallback(csv_path=os.path.join(alg_folder, "curriculum_metrics.csv"))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=alg_folder,
        log_path=os.path.join(alg_folder, "eval_logs"),
        eval_freq=max(1, total_timesteps // 10),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0
    )

    # Crear modelo SAC
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        ent_coef=ent_coef,
        verbose=0,
        tensorboard_log=os.path.join(alg_folder, "tb_logs")
    )
    model.set_logger(logger)

    # Entrenar con barra de progreso integrada
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([curriculum_cb, eval_cb]),
        progress_bar=True
    )

    # Guardar modelo final
    model.save(os.path.join(alg_folder, "final_model"))

    # Evaluar recompensa media final
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    return mean_reward


def run_hpo(alg_name: str, env_kwargs: dict, total_timesteps: int, n_trials: int):
    """
    Ejecuta HPO con Optuna para 'PPO' o 'SAC'.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    study = optuna.create_study(
        sampler=TPESampler(),
        direction="maximize",
        study_name=f"{alg_name}_Study_{datetime.now():%Y%m%d_%H%M%S}"
    )

    if alg_name == "PPO":
        func = lambda trial: optimize_ppo(trial, env_kwargs, total_timesteps, RESULTS_DIR)
    else:
        func = lambda trial: optimize_sac(trial, env_kwargs, total_timesteps, RESULTS_DIR)

    study.optimize(func, n_trials=n_trials)

    # Guardar resultados del estudio
    df = study.trials_dataframe()
    df.to_csv(os.path.join(RESULTS_DIR, f"{alg_name}_hpo_results.csv"), index=False)

    print(f"\n=== HPO {alg_name} completado ===")
    print("Mejores parámetros:", study.best_params)
    print(f"Recompensa media objetivo: {study.best_value:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento HPO con SB3 para PepperArmEnv")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Timesteps por trial")
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials de HPO por algoritmo")
    parser.add_argument("--side", choices=["Left", "Right"], default="Left",
                        help="Brazo a entrenar")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Muestras por dimensión en CSpace")
    parser.add_argument("--start_frac", type=float, default=0.2,
                        help="Frac. inicial del currículo")
    parser.add_argument("--incr_frac", type=float, default=0.1,
                        help="Frac. de incremento del currículo")
    parser.add_argument("--max_steps", type=int, default=250,
                        help="Máx. pasos por episodio")
    parser.add_argument("--required_succ", type=int, default=2,
                        help="Éxitos consecutivos para subir nivel")
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

    print("\n=== Iniciando HPO ===")
    print(f"Configuración de entorno: {env_kwargs}")
    print(f"Timesteps: {args.timesteps}, Trials: {args.trials}\n")

    #run_hpo("PPO", env_kwargs, args.timesteps, args.trials)
    run_hpo("SAC", env_kwargs, args.timesteps, args.trials)

    print("=== Entrenamiento y HPO completados ===")
