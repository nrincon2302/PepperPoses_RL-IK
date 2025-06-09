# model_statistics_graphs.py

import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_plot(model_name: str):
    base = "resultados_calibracion"
    folder = os.path.join(base, model_name)

    # Paths
    monitor_csv = os.path.join(folder, "monitor.csv")
    progress_csv = os.path.join(folder, "tb_logs", "progress.csv")
    curr_csv = os.path.join(folder, "curriculum_metrics.csv")

    # 1) Métricas globales
    print(f"\n=== Métricas globales de '{model_name}' ===")
    if os.path.exists(monitor_csv):
        df_mon = pd.read_csv(monitor_csv)
        # Detectar columnas
        rew_col = next((c for c in df_mon.columns if 'reward' in c.lower() or c == 'r'), None)
        len_col = next((c for c in df_mon.columns if 'l' == c), None)

        if rew_col:
            rewards = df_mon[rew_col]
            print(f"Total episodios: {len(rewards)}")
            print(f"  Recompensa media   : {rewards.mean():.2f}")
            print(f"  Recompensa mediana : {rewards.median():.2f}")
            print(f"  Desv. estándar     : {rewards.std():.2f}")
        if len_col:
            lengths = df_mon[len_col]
            print(f"  Longitud media ep. : {lengths.mean():.1f} pasos")
        # Éxito global si is_success en curriculum
    else:
        print(f"⚠️ monitor.csv no encontrado en {folder}")

    # 2) Evolución recompensa + value_loss
    fig, ax1 = plt.subplots()
    if os.path.exists(monitor_csv) and rew_col:
        ax1.plot(df_mon.index + 1, df_mon[rew_col], color='tab:blue', label='Reward per episode')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    if os.path.exists(progress_csv):
        df_tb = pd.read_csv(progress_csv)
        # Detectar columnas
        valcol = next((c for c in df_tb.columns if 'value_loss' in c.lower()), None)
        itcol = next((c for c in df_tb.columns if 'iter' in c.lower() or 'step' in c.lower()), df_tb.columns[0])
        if valcol:
            ax2.plot(df_tb[itcol], df_tb[valcol], color='tab:red', label='Value loss')
            ax2.set_ylabel("Value Loss", color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.suptitle(f"{model_name}: Recompensa y Value Loss")
    fig.tight_layout()
    plt.show()

    # 3) Estadísticas de recompensa por etapa del currículo
    if os.path.exists(curr_csv):
        df_curr = pd.read_csv(curr_csv)
        # Agrupar por etapa: detectamos los cambios en curriculum_radius
        df_curr['stage'] = (df_curr['curriculum_radius']
                            .diff().fillna(0) > 0).cumsum() + 1
        stages = df_curr['stage'].unique()

        # Boxplot de recompensas por stage
        plt.figure(figsize=(8, 4))
        data = [df_curr.loc[df_curr['stage'] == s, 'total_reward'] for s in stages]
        plt.boxplot(data, labels=[f"S{s}" for s in stages])
        plt.xlabel("Curriculum Stage")
        plt.ylabel("Recompensa por episodio")
        plt.title(f"{model_name}: Distribución de recompensa por etapa")
        plt.tight_layout()
        plt.show()

        # Barra de recompensa media por stage
        means = [d.mean() for d in data]
        plt.figure()
        plt.bar([f"S{s}" for s in stages], means)
        plt.xlabel("Curriculum Stage")
        plt.ylabel("Recompensa media")
        plt.title(f"{model_name}: Recompensa media por etapa")
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ curriculum_metrics.csv no encontrado en {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graficar métricas avanzadas de entrenamiento"
    )
    parser.add_argument(
        "model_name",
        help="Nombre de la carpeta del experimento (e.g. PPO-1)"
    )
    args = parser.parse_args()
    analyze_and_plot(args.model_name)
