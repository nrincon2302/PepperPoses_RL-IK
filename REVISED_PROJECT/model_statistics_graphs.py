import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_stats(model_name: str):
    base = "resultados_calibracion"
    folder = os.path.join(base, model_name)

    # 1) monitor.csv
    df_mon = pd.read_csv(os.path.join(folder, "monitor.csv"), comment='#')
    # columnas: r,l,t

    # 2) curriculum_metrics.csv
    df_curr = pd.read_csv(os.path.join(folder, "curriculum_metrics.csv"))

    # 3) progress.csv con cabeceras según algoritmo
    prog_path = os.path.join(folder, "tb_logs", "progress.csv")
    df_tb = pd.read_csv(prog_path)

    alg = model_name.split("-")[0]
    if alg not in ["PPO", "SAC"]:
        raise ValueError(f"Algoritmo no soportado: {alg}. Debe ser 'PPO' o 'SAC'.")

    if alg == "PPO":
        # columnas PPO
        cols = [
            "time/time_elapsed","time/iterations","rollout/ep_len_mean","rollout/success_rate",
            "time/fps","rollout/ep_rew_mean","time/total_timesteps","train/n_updates",
            "train/clip_range","train/learning_rate","train/value_loss",
            "train/approx_kl","train/explained_variance","train/clip_fraction",
            "train/std","train/policy_gradient_loss","train/entropy_loss","train/loss"
        ]
    else:
        # columnas SAC
        cols = [
            "rollout/ep_rew_mean","rollout/success_rate","train/ent_coef","train/critic_loss",
            "time/time_elapsed","train/actor_loss","rollout/ep_len_mean","train/n_updates",
            "time/total_timesteps","train/learning_rate","time/fps","time/episodes"
        ]

    # Filtrar solo las columnas existentes
    cols = [c for c in cols if c in df_tb.columns]

    # 1) Reward & Success Rate
    plt.figure(figsize=(8,4))
    eps = np.arange(1, len(df_mon)+1)
    plt.plot(eps, df_mon["r"], label="Recompensa", color='tab:blue')
    sr = df_curr["is_success"].rolling(10, min_periods=1).mean()
    plt.plot(df_curr["episode"], sr * df_mon["r"].max(), label="Éxito(rolling)", color='tab:green')
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title(f"{model_name}: Reward & Success")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Training metrics
    plt.figure(figsize=(8,4))
    if "rollout/ep_rew_mean" in cols:
        plt.plot(df_tb["time/total_timesteps"], df_tb["rollout/ep_rew_mean"], label="Rew rollout")
    if "train/value_loss" in cols:
        plt.plot(df_tb["time/total_timesteps"], df_tb["train/value_loss"], label="Value loss")
    if "train/actor_loss" in cols:
        plt.plot(df_tb["time/total_timesteps"], df_tb["train/actor_loss"], label="Actor loss")
    if "train/critic_loss" in cols:
        plt.plot(df_tb["time/total_timesteps"], df_tb["train/critic_loss"], label="Critic loss")
    plt.xlabel("Timesteps")
    plt.ylabel("Valor / Loss")
    plt.title(f"{model_name}: Training Metrics ({alg})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Curriculum progress
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df_curr["episode"], df_curr["curriculum_radius"], color='tab:orange', label="Radio")
    ax2 = ax1.twinx()
    ax2.plot(df_curr["episode"], df_curr["consecutive_successes"], color='tab:purple', label="Éxitos seg.")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Radio [m]")
    ax2.set_ylabel("Éxitos")
    fig.suptitle(f"{model_name}: Curriculum Progress")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # 4) Reward / stage
    df_curr['stage'] = (df_curr['curriculum_radius'].diff().fillna(0)>0).cumsum()+1
    stats = df_curr.groupby('stage')['total_reward'].agg(['mean','std'])
    plt.figure(figsize=(6,4))
    x = stats.index.astype(str)
    plt.bar(x, stats['mean'], yerr=stats['std'], capsize=5)
    plt.xlabel("Stage"); plt.ylabel("Reward")
    plt.title(f"{model_name}: Reward per Stage")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_name")
    args = p.parse_args()
    plot_stats(args.model_name)
