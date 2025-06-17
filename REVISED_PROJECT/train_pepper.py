import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

import optuna
from optuna.samplers import TPESampler

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure

from pepper_env import PepperArmEnv

RESULTS_DIR = "resultados_calibracion"


class CurriculumCallback(BaseCallback):
    """
    Callback para gestionar el Aprendizaje por Currículo y registrar métricas.
    """
    def __init__(self, curriculum_params: dict, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        
        # Parámetros del currículo
        self.start_frac = curriculum_params["start_frac"]
        self.increment_frac = curriculum_params["increment_frac"]
        self.required_successes = curriculum_params["required_successes"]
        
        # Estado del currículo
        self.consecutive_successes = 0
        self.current_radius = 0.0
        self.max_radius = 0.0
        self.increment = 0.0
        self.initialized = False
        
        # Logging
        self.episode_count = 0
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=[
                "episode", "total_reward", "episode_length", "is_success",
                "curriculum_radius", "consecutive_successes"
            ]).to_csv(self.csv_path, index=False)


    def _on_training_start(self) -> None:
        """
        Inicializa los parámetros del currículo usando el entorno.
        """
        if not self.initialized:
            env = self.training_env.envs[0].env # Acceder al entorno base
            self.max_radius = env.max_workspace_radius
            self.current_radius = self.start_frac * self.max_radius
            self.increment = self.increment_frac * self.max_radius
            env.set_curriculum_radius(self.current_radius)
            self.initialized = True
            # Escritura en consola del entrenamiento inicializado
            if self.verbose > 0:
                print("--- Callback del Curriculum iniciado ---")
                print(f"Radio máximo: {self.max_radius:.3f} m")
                print(f"Radio Inicial: {self.current_radius:.3f} m")
                print(f"Incremento en el Radio: {self.increment:.3f} m")
                print("---------------------------------------")

    def _on_step(self) -> bool:
        """
        Lógica del currículo después de cada paso.
        Actualiza el radio del currículo y registra métricas.
        """
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                info = self.locals["infos"][i]
                
                # ===============================================
                # Actualización del currículo
                # ===============================================
                # Contar la cantidad de éxitos consecutivos
                if info.get("is_success", False):
                    self.consecutive_successes += 1
                else:
                    self.consecutive_successes = 0
                
                # Si se alcanzó el número requerido de éxitos, aumentar el radio
                if self.consecutive_successes >= self.required_successes:
                    self.current_radius = min(self.current_radius + self.increment, self.max_radius)
                    self.consecutive_successes = 0 # Resetear a cero
                    if self.verbose > 0:
                        print(f"\n[Curriculum] Nivel aumentado! Nuevo radio: {self.current_radius:.3f} m\n")

                # Actualizar el radio en el entorno para el próximo reset
                self.training_env.env_method("set_curriculum_radius", self.current_radius)

                # ===============================================
                # Registro de métricas
                # ===============================================
                if "episode" in info:
                    epi_info = info["episode"]
                    log_data = {
                        "episode": self.episode_count,
                        "total_reward": epi_info["r"],
                        "episode_length": epi_info["l"],
                        "is_success": int(info.get("is_success", False)),
                        "curriculum_radius": self.current_radius,
                        "consecutive_successes": self.consecutive_successes,
                    }
                    df = pd.DataFrame([log_data])
                    df.to_csv(self.csv_path, mode='a', header=False, index=False)

                self.logger.record("curriculum/radius", self.current_radius)
                self.logger.record("curriculum/consecutive_successes", self.consecutive_successes)
                if 'is_success' in info:
                    self.logger.record("rollout/success_rate", info.get('is_success', False))

        return True

def make_env(seed: int, log_folder: str, **env_kwargs):
    """
    Crea un entorno PepperArmEnv con monitorización y logging.
    """
    os.makedirs(log_folder, exist_ok=True)
    env_raw = PepperArmEnv(**env_kwargs)
    monitor_path = os.path.join(log_folder, "monitor.csv")
    return Monitor(env_raw, filename=monitor_path)


def optimize_agent(trial: optuna.Trial, alg_name: str, env_kwargs: dict, curriculum_params: dict, total_timesteps: int, base_dir: str):
    """
    Función objetivo genérica para PPO y SAC.
    """
    trial_id = trial.number
    alg_folder = os.path.join(base_dir, f"{alg_name}-{trial_id}")
    os.makedirs(alg_folder, exist_ok=True)
    
    # Logger de SB3
    logger = configure(os.path.join(alg_folder, "tb_logs"), ["stdout", "csv", "tensorboard"])
    tb_log_path = os.path.join(alg_folder, "tb_logs")
    
    # Entornos
    train_env = make_env(seed=trial_id, log_folder=alg_folder, **env_kwargs)
    eval_env = make_env(seed=trial_id + 1000, log_folder=os.path.join(alg_folder, "eval"), **env_kwargs)
    
    # Callbacks
    curriculum_cb = CurriculumCallback(
        curriculum_params=curriculum_params,
        csv_path=os.path.join(alg_folder, "curriculum_metrics.csv"),
        verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=alg_folder,
        log_path=os.path.join(alg_folder, "eval_logs"),
        eval_freq=max(5000, total_timesteps // 20),
        n_eval_episodes=10,
        deterministic=True
    )

    # Definir la arquitectura de la política
    # Usamos una red MLP con 2 capas ocultas de 256 neuronas cada una
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    # Hiperparámetros y creación del modelo
    if alg_name == "PPO":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "batch_size": 64,
        }
        model = PPO("MlpPolicy", train_env, **params, policy_kwargs=policy_kwargs, tensorboard_log=tb_log_path)
    else:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 300_000, 1_000_000]),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "tau": trial.suggest_float("tau", 0.005, 0.05),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4, 8, 16]),
            "ent_coef": 'auto',
        }
        model = SAC("MlpPolicy", train_env, **params, policy_kwargs=policy_kwargs, tensorboard_log=tb_log_path)
    
    model.set_logger(logger)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([curriculum_cb, eval_cb]),
            progress_bar=True
        )
    except Exception as e:
        print(f"Error durante el entrenamiento en el trial {trial_id}: {e}")
        raise optuna.exceptions.TrialPruned()

    # Guardar modelo final y evaluar
    model.save(os.path.join(alg_folder, "final_model"))
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1000)
    
    # Cerrar entornos para liberar recursos de pybullet
    train_env.close()
    eval_env.close()

    return mean_reward


def evaluate_policy(model, env, n_eval_episodes: int = 10):
    """
    Evalúa la política entrenada en el entorno dado.
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

def run_hpo(alg_name: str, env_kwargs: dict, curriculum_params: dict, total_timesteps: int, n_trials: int):
    """
    Ejecuta la optimización de hiperparámetros (HPO) para el algoritmo especificado.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        direction="maximize",
        study_name=f"{alg_name}_Study_{datetime.now():%Y%m%d_%H%M%S}"
    )

    func = lambda trial: optimize_agent(trial, alg_name, env_kwargs, curriculum_params, total_timesteps, RESULTS_DIR)
    
    try:
        study.optimize(func, n_trials=n_trials, timeout=3600*24) # 24 horas de timeout
    except KeyboardInterrupt:
        print("HPO interrumpido por el usuario.")

    # Guardar resultados
    df = study.trials_dataframe()
    df.to_csv(os.path.join(RESULTS_DIR, f"{alg_name}_hpo_results.csv"), index=False)

    print(f"\n=== HPO {alg_name} completado ===")
    print("Mejores parámetros:", study.best_params)
    print(f"Recompensa media objetivo: {study.best_value:.2f}\n")


# =========================================================
# Punto de entrada principal
# =========================================================
# Permite ejecutar el script desde la línea de comandos con argumentos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento HPO con SB3 para PepperArmEnv")
    parser.add_argument("--alg", choices=["PPO", "SAC"], default="SAC", help="Algoritmo a optimizar")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Timesteps por trial")
    parser.add_argument("--trials", type=int, default=5, help="Trials de HPO")
    parser.add_argument("--side", choices=["Left", "Right"], default="Left", help="Brazo a entrenar")
    parser.add_argument("--n_samples", type=int, default=8, help="Muestras por dimensión en CSpace")
    parser.add_argument("--max_steps", type=int, default=250, help="Máx. pasos por episodio")
    # Argumentos del currículo
    parser.add_argument("--start_frac", type=float, default=0.2, help="Frac. inicial del currículo")
    parser.add_argument("--incr_frac", type=float, default=0.1, help="Frac. de incremento del currículo")
    parser.add_argument("--required_succ", type=int, default=5, help="Éxitos consecutivos para subir nivel")

    args = parser.parse_args()

    env_kwargs = {
        "side": args.side,
        "render_mode": None, # Sin GUI para acelerar el entrenamiento y el estudio HPO
        "max_steps": args.max_steps,
        "n_workspace_samples": args.n_samples,
    }

    args = parser.parse_args()

    curriculum_params = {
        "start_frac": args.start_frac,
        "increment_frac": args.incr_frac,
        "required_successes": args.required_succ,
    }

    print(f"\n=== Iniciando HPO para {args.alg} ===")
    print(f"Configuración de entorno: {env_kwargs}")
    print(f"Parámetros de currículo: {curriculum_params}")
    print(f"Timesteps: {args.timesteps}, Trials: {args.trials}\n")

    run_hpo(args.alg, env_kwargs, curriculum_params, args.timesteps, args.trials)

    print("=== Entrenamiento y HPO completados ===")
