"""
Script para entrenar PepperArmEnv usando Stable Baselines 3 (PPO).
Simplemente ejecútalo con: python train_pepper_arm_sb3.py
"""

import os

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pepper_env import PepperArmEnv

# ------------------------------------------------------------
# CONFIGURACIÓN FIJA PARA NO USAR ARGUMENTOS DE CONSOLA:
# ------------------------------------------------------------
TOTAL_TIMESTEPS = 500_000
LOG_DIR        = "./logs/"
SAVE_PATH      = "./models/ppo_pepper_arm.zip"

# Número de entornos paralelos para evaluación
EVAL_ENVS      = 4
# Episodios de evaluación cada vez que se evalúa
EVAL_EPISODES  = 5
# Umbral de recompensa media para detener antes
REWARD_THRESHOLD = 200.0

# Frecuencias de callbacks (en pasos)
CHECKPOINT_FREQ = 50_000   # cada 50k pasos, guardamos checkpoint
EVAL_FREQ       = 100_000  # cada 100k pasos, evaluamos
# ------------------------------------------------------------


def make_env(side: str, render_mode: str = None):
    """
    Factory para crear instancias de PepperArmEnv envuelto en Monitor.
    """
    def _init():
        env = PepperArmEnv(
            side=side,
            render_mode=render_mode,
            max_steps=250,
            n_workspace_samples=8,
            curriculum_start_frac=0.2,
            curriculum_increment_frac=0.1
        )
        env = Monitor(env)
        return env

    return _init


def main():
    # Crear carpetas si no existen
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ------------------ 1. Crear entorno(es) de entrenamiento ------------------
    train_env = DummyVecEnv([make_env(side="Left", render_mode=None)])
    # (Opcional) Normalizar observaciones/recompensas:
    # from stable_baselines3.common.vec_env import VecNormalize
    # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ------------------ 2. Crear entorno(es) de evaluación ------------------
    eval_env_fns = [make_env(side="Left", render_mode=None) for _ in range(EVAL_ENVS)]
    eval_env = DummyVecEnv(eval_env_fns)
    # Si usas VecNormalize en entrenamiento, descomenta y ajusta:
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # eval_env.training = False
    # eval_env.norm_reward = False

    # ------------------ 3. Definir CALLBACKS ------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.dirname(SAVE_PATH),
        name_prefix="pepper_arm_ppo_checkpoint"
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLD,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        n_eval_episodes=EVAL_EPISODES,
        best_model_save_path=os.path.dirname(SAVE_PATH),
        log_path=os.path.join(LOG_DIR, "eval_logs"),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False
    )

    # ------------------ 4. Crear modelo PPO ------------------
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard/"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    # ------------------ 5. Entrenamiento ------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )

    # ------------------ 6. Guardar modelo final ------------------
    model.save(SAVE_PATH)
    print(f"\nModelo final guardado en: {SAVE_PATH}")


if __name__ == "__main__":
    main()