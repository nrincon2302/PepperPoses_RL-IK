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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

# Importamos ambos tipos de entorno para poder elegirlos dinámicamente
from environments.pepper_env import PepperArmEnv
from environments.pepper_analytical_env import PepperAnalyticalEnv

RESULTS_DIR = "resultados_calibracion"


def make_env_fn(log_folder: str, env_type: str, **env_params):
    """
    Función de ayuda para crear el entorno correcto (simulado o analítico) con un Monitor.
    """
    def _init():
        os.makedirs(log_folder, exist_ok=True)
        # Elige la clase del entorno a instanciar
        EnvClass = PepperArmEnv if env_type == "sim" else PepperAnalyticalEnv
        
        env_raw = EnvClass(**env_params)
        monitor_path = os.path.join(log_folder, "monitor.csv")
        return Monitor(env_raw, filename=monitor_path)
    return _init


class CurriculumCallback(BaseCallback):
    """
    Callback para gestionar el Aprendizaje por Currículo y registrar métricas.
    """
    def __init__(self, curriculum_params: dict, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.start_frac = curriculum_params["start_frac"]
        self.increment_frac = curriculum_params["increment_frac"]
        self.required_successes = curriculum_params["required_successes"]
        self.consecutive_successes = 0
        self.current_radius = 0.0
        self.max_radius = 0.0
        self.increment = 0.0
        self.initialized = False
        self.episode_count = 0
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=[
                "episode", "total_reward", "episode_length", "is_success",
                "curriculum_radius", "consecutive_successes"
            ]).to_csv(self.csv_path, index=False)

    def _on_training_start(self) -> None:
        if not self.initialized:
            all_max_radii = self.training_env.get_attr('max_workspace_radius')
            self.max_radius = all_max_radii[0]
            self.current_radius = self.start_frac * self.max_radius
            self.increment = self.increment_frac * self.max_radius
            self.training_env.env_method("set_curriculum_radius", self.current_radius)
            self.initialized = True
            if self.verbose > 0:
                print(f"--- Curriculum iniciado: radio inicial {self.current_radius:.3f} m, incremento {self.increment:.3f} m ---")

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                info = self.locals["infos"][i]
                if info.get("is_success", False):
                    self.consecutive_successes += 1
                else:
                    self.consecutive_successes = 0
                if self.consecutive_successes >= self.required_successes:
                    self.current_radius = min(self.current_radius + self.increment, self.max_radius)
                    self.consecutive_successes = 0
                    if self.verbose > 0:
                        print(f"\n[Curriculum] Nivel aumentado: nuevo radio -> {self.current_radius:.3f} m\n")
                self.training_env.env_method("set_curriculum_radius", self.current_radius)
                if "episode" in info:
                    epi = info["episode"]
                    pd.DataFrame([{"episode": self.episode_count, "total_reward": epi["r"], "episode_length": epi["l"], "is_success": int(info.get("is_success", False)), "curriculum_radius": self.current_radius, "consecutive_successes": self.consecutive_successes}]).to_csv(self.csv_path, mode='a', header=False, index=False)
                self.logger.record("curriculum/radius", self.current_radius)
                self.logger.record("rollout/success_rate", info.get('is_success', False))
        return True


def optimize_agent(trial: optuna.Trial, alg_name: str, env_type: str, env_params: dict, curriculum_params: dict, total_timesteps: int, n_envs: int, base_dir: str):
    trial_id = trial.number
    alg_folder = os.path.join(base_dir, f"{alg_name}-{env_type}-{trial_id}")
    os.makedirs(alg_folder, exist_ok=True)
    logger = configure(os.path.join(alg_folder, "tb_logs"), ["stdout", "csv", "tensorboard"])

    # El entorno analítico es muy rápido, por lo que DummyVecEnv (secuencial) es mejor.
    # El entorno de simulación es lento, por lo que SubprocVecEnv (paralelo, por defecto con None) es mejor.
    vec_env_cls = DummyVecEnv if env_type == "analytical" else None

    train_env = make_vec_env(
        make_env_fn(log_folder=alg_folder, env_type=env_type, **env_params),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls
    )
    eval_env = make_vec_env(
        make_env_fn(log_folder=os.path.join(alg_folder, "eval"), env_type=env_type, **env_params),
        n_envs=1
    )
    curriculum_cb = CurriculumCallback(curriculum_params=curriculum_params, csv_path=os.path.join(alg_folder, "curriculum_metrics.csv"), verbose=1)
    eval_cb = EvalCallback(eval_env, best_model_save_path=alg_folder, log_path=os.path.join(alg_folder, "eval_logs"), eval_freq=max(5000, total_timesteps // (20 * n_envs)), n_eval_episodes=20, deterministic=True, warn=False)

    if alg_name == "PPO":
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        hyperparams = {'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True), 'n_steps': trial.suggest_categorical('n_steps', [256, 512, 1024, 2048]), 'gamma': trial.suggest_float('gamma', 0.9, 0.9999), 'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99), 'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True), 'clip_range': trial.suggest_float('clip_range', 0.1, 0.4), 'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0), 'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256])}
        model = PPO('MlpPolicy', train_env, policy_kwargs=policy_kwargs, tensorboard_log=None, **hyperparams)
    else:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        hyperparams = {'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True), 'buffer_size': trial.suggest_categorical('buffer_size', [100000, 300000, 1000000]), 'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]), 'tau': trial.suggest_float('tau', 0.005, 0.05), 'gamma': trial.suggest_float('gamma', 0.9, 0.9999), 'train_freq': (trial.suggest_categorical('train_freq', [1,4,8,16]), 'step'), 'gradient_steps': trial.suggest_categorical('gradient_steps', [-1,1,4,8,16]), 'ent_coef': 'auto'}
        model = SAC('MlpPolicy', train_env, policy_kwargs=policy_kwargs, tensorboard_log=None, **hyperparams)
    
    model.set_logger(logger)
    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList([curriculum_cb, eval_cb]), progress_bar=True)
    except Exception as e:
        print(f"Trial {trial_id} falló: {e}")
        raise optuna.TrialPruned()
    
    model.save(os.path.join(alg_folder, 'final_model'))
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=100)
    train_env.close()
    eval_env.close()
    return mean_reward


def evaluate_policy(model, env: VecEnv, n_eval_episodes: int = 10):
    """
    Evalúa la política entrenada en el entorno vectorizado dado.
    """
    rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = [False]
        total_r = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            # El step de un VecEnv devuelve 4 valores
            obs, r, done, info = env.step(action)
            total_r += r[0] # La recompensa 'r' es un array
        rewards.append(total_r)
    return np.mean(rewards), np.std(rewards)


def run_hpo(alg_name: str, env_type: str, env_params: dict, curriculum_params: dict, total_timesteps: int, n_trials: int, n_envs: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    study_name = f"{alg_name}_{env_type}_Study_{datetime.now():%Y%m%d_%H%M%S}"
    study = optuna.create_study(sampler=TPESampler(seed=42), direction='maximize', study_name=study_name)
    
    func = lambda trial: optimize_agent(trial, alg_name, env_type, env_params, curriculum_params, total_timesteps, n_envs, RESULTS_DIR)
    
    try:
        study.optimize(func, n_trials=n_trials)
    except KeyboardInterrupt:
        print("HPO interrumpido por el usuario.")

    df = study.trials_dataframe()
    df.to_csv(os.path.join(RESULTS_DIR, f"{study_name}_results.csv"), index=False)
    
    print(f"\n=== HPO {alg_name} ({env_type}) completado ===")
    if study.best_trial:
        print("Mejores parámetros:", study.best_params)
        print(f"Recompensa media objetivo: {study.best_value:.2f}\n")
    else:
        print("No se completó ningún trial exitosamente.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento HPO con SB3 para el brazo de Pepper')
    parser.add_argument("--env_type", choices=["sim", "analytical"], default="analytical", help="Tipo de entorno a usar: 'sim' para PyBullet, 'analytical' para el modelo matemático rápido.")
    parser.add_argument('--alg', choices=['PPO','SAC'], default='PPO', help='Algoritmo de RL a usar.')
    parser.add_argument('--timesteps', type=int, default=500000, help='Número de timesteps de entrenamiento por trial de HPO.')
    parser.add_argument('--trials', type=int, default=5, help='Número de trials de HPO a ejecutar.')
    parser.add_argument('--side', choices=['Left','Right'], default='Left', help='Brazo del robot a entrenar.')
    parser.add_argument('--n_envs', type=int, default=4, help='Número de entornos paralelos a usar.')
    parser.add_argument('--n_samples', type=int, default=8, help='Resolución del muestreo del workspace (más alto = más denso pero más lento al inicio).')
    parser.add_argument('--max_steps', type=int, default=250, help='Máximo número de pasos por episodio.')
    # Argumentos del currículo
    parser.add_argument('--start_frac', type=float, default=0.2, help='Fracción inicial del radio del workspace para el currículo.')
    parser.add_argument('--incr_frac', type=float, default=0.1, help='Fracción de incremento del radio para el currículo.')
    parser.add_argument('--required_succ', type=int, default=5, help='Éxitos consecutivos para aumentar el nivel del currículo.')
    args = parser.parse_args()

    # Construye el diccionario de parámetros para el entorno de forma limpia
    env_params = {
        'side': args.side,
        'max_steps': args.max_steps,
        'n_workspace_samples': args.n_samples,
    }
    # El entorno simulado necesita render_mode, pero para entrenamiento siempre es None (sin GUI)
    if args.env_type == "sim":
        env_params['render_mode'] = None 

    curriculum_params = {
        'start_frac': args.start_frac,
        'increment_frac': args.incr_frac,
        'required_successes': args.required_succ,
    }
    
    print(f"\n=== Iniciando HPO para {args.alg} en entorno '{args.env_type}' ===")
    print(f"Usando {args.n_envs} entornos paralelos.")
    print(f"Parámetros de entorno: {env_params}")

    run_hpo(
        alg_name=args.alg,
        env_type=args.env_type,
        env_params=env_params,
        curriculum_params=curriculum_params,
        total_timesteps=args.timesteps,
        n_trials=args.trials,
        n_envs=args.n_envs
    )
    
    print('=== Entrenamiento y HPO completados ===')