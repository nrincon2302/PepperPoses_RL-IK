# scripts/plot_training_curve.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def plot_monitor_log(monitor_csv_path, window=100, output_dir="."):
    """
    Plotea la recompensa media móvil desde el archivo monitor.csv de SB3.
    """
    if not os.path.exists(monitor_csv_path):
        print(f"Archivo monitor no encontrado: {monitor_csv_path}")
        return

    try:
        df = pd.read_csv(monitor_csv_path, skiprows=1) # Saltar la primera fila de cabecera de SB3
    except Exception as e:
        print(f"Error leyendo {monitor_csv_path}: {e}")
        return
        
    if 'r' not in df.columns:
        print(f"Columna 'r' (reward) no encontrada en {monitor_csv_path}")
        return

    plt.figure(figsize=(12, 6))
    df['r_smooth'] = df['r'].rolling(window=window, min_periods=1).mean()
    plt.plot(df.index, df['r'], alpha=0.3, label=f'Recompensa Episodio')
    plt.plot(df.index, df['r_smooth'], label=f'Recompensa Media Móvil (ventana={window})')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Total del Episodio')
    plt.title('Curva de Entrenamiento - Recompensa del Episodio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_reward_curve.png"))
    print(f"Gráfica de recompensa guardada en {os.path.join(output_dir, 'training_reward_curve.png')}")
    # plt.show() # Descomentar si quieres mostrar interactivamente


def plot_curriculum_stats(detailed_stats_csv_path, aggregated_stats_csv_path, output_dir="."):
    """
    Plotea estadísticas por nivel de currículo desde los CSVs generados por el callback.
    """
    if not os.path.exists(detailed_stats_csv_path):
        print(f"Archivo de estadísticas detalladas del currículo no encontrado: {detailed_stats_csv_path}")
        return
    if not os.path.exists(aggregated_stats_csv_path):
        print(f"Archivo de estadísticas agregadas del currículo no encontrado: {aggregated_stats_csv_path}")
        # Podríamos intentar calcularlo desde el detallado si solo ese existe
        # return 

    try:
        df_detailed = pd.read_csv(detailed_stats_csv_path)
    except Exception as e:
        print(f"Error leyendo {detailed_stats_csv_path}: {e}")
        df_detailed = None # Marcar para no usarlo si falla

    try:
        df_agg = pd.read_csv(aggregated_stats_csv_path)
    except Exception as e:
        print(f"Error leyendo {aggregated_stats_csv_path}: {e}")
        # Si el agregado falla, pero el detallado existe, podemos intentar agregarlo aquí
        if df_detailed is not None and not df_detailed.empty:
            print("Intentando agregar estadísticas desde el archivo detallado.")
            df_agg = df_detailed.groupby('curriculum_level').agg(
                mean_reward=('reward', 'mean'),
                std_reward=('reward', 'std'),
                mean_length=('length', 'mean'),
                success_rate=('success', 'mean'),
                mean_final_distance=('final_distance', 'mean'),
                episodes_in_level=('reward', 'count')
            ).reset_index()
        else:
            print("No se pueden generar gráficos de currículo agregados.")
            return # No podemos continuar sin datos agregados

    if df_agg.empty:
        print("No hay datos agregados del currículo para graficar.")
        return

    # Gráficas usando datos agregados
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # 1. Recompensa media por nivel de currículo
    axs[0].plot(df_agg['curriculum_level'], df_agg['mean_reward'], marker='o', linestyle='-')
    if 'std_reward' in df_agg.columns: # Si tenemos std_reward
        axs[0].fill_between(df_agg['curriculum_level'], 
                            df_agg['mean_reward'] - df_agg['std_reward'], 
                            df_agg['mean_reward'] + df_agg['std_reward'], 
                            alpha=0.2)
    axs[0].set_ylabel('Recompensa Media')
    axs[0].set_title('Recompensa Media por Nivel de Currículo')
    axs[0].grid(True)

    # 2. Tasa de éxito por nivel de currículo
    if 'success_rate' in df_agg.columns:
        axs[1].plot(df_agg['curriculum_level'], df_agg['success_rate'] * 100, marker='o', linestyle='-')
        axs[1].set_ylabel('Tasa de Éxito (%)')
        axs[1].set_title('Tasa de Éxito por Nivel de Currículo')
        axs[1].grid(True)
    else:
        axs[1].text(0.5, 0.5, 'Datos de tasa de éxito no disponibles', ha='center', va='center')


    # 3. Distancia final media por nivel de currículo
    if 'mean_final_distance' in df_agg.columns:
        axs[2].plot(df_agg['curriculum_level'], df_agg['mean_final_distance'], marker='o', linestyle='-')
        axs[2].set_ylabel('Distancia Final Media (m)')
        axs[2].set_title('Distancia Final Media por Nivel de Currículo')
        axs[2].grid(True)
    else:
        axs[2].text(0.5, 0.5, 'Datos de distancia final no disponibles', ha='center', va='center')
        
    axs[2].set_xlabel('Nivel de Currículo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curriculum_performance_summary.png"))
    print(f"Gráfica de resumen del currículo guardada en {os.path.join(output_dir, 'curriculum_performance_summary.png')}")
    # plt.show() # Descomentar si quieres mostrar interactivamente

    # Opcional: Boxplot de recompensas por nivel de currículo usando datos detallados
    if df_detailed is not None and not df_detailed.empty and 'reward' in df_detailed.columns:
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='curriculum_level', y='reward', data=df_detailed)
        plt.title('Distribución de Recompensas por Nivel de Currículo')
        plt.xlabel('Nivel de Currículo')
        plt.ylabel('Recompensa del Episodio')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "curriculum_reward_distribution.png"))
        print(f"Gráfica de distribución de recompensas del currículo guardada en {os.path.join(output_dir, 'curriculum_reward_distribution.png')}")
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genera gráficas de entrenamiento y currículo.")
    parser.add_argument("run_dir", type=str, help="Directorio del run de entrenamiento específico (ej: training_results_pepper/ppo_pepper5dof_TIMESTAMP)")
    parser.add_argument("--monitor_file", type=str, default="monitor_logs/monitor.csv", help="Nombre relativo del archivo monitor CSV dentro de run_dir/monitor_logs.")
    parser.add_argument("--detailed_stats_file", type=str, default="detailed_curriculum_episode_stats.csv", help="Nombre del archivo CSV de estadísticas detalladas del currículo dentro de run_dir.")
    parser.add_argument("--aggregated_stats_file", type=str, default="aggregated_curriculum_stats.csv", help="Nombre del archivo CSV de estadísticas agregadas del currículo dentro de run_dir.")
    parser.add_argument("--plot_output_dir", type=str, default=None, help="Directorio para guardar las gráficas. Por defecto, el mismo run_dir.")


    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: El directorio del run '{args.run_dir}' no existe.")
        exit(1)

    monitor_path = os.path.join(args.run_dir, args.monitor_file)
    detailed_stats_path = os.path.join(args.run_dir, args.detailed_stats_file)
    aggregated_stats_path = os.path.join(args.run_dir, args.aggregated_stats_file)
    
    plot_output_directory = args.plot_output_dir if args.plot_output_dir else args.run_dir
    os.makedirs(plot_output_directory, exist_ok=True)

    print(f"Generando gráficas para el run en: {args.run_dir}")
    print(f"Guardando gráficas en: {plot_output_directory}")

    plot_monitor_log(monitor_path, output_dir=plot_output_directory)
    plot_curriculum_stats(detailed_stats_path, aggregated_stats_path, output_dir=plot_output_directory)
    
    print("\nProceso de ploteo completado.")
    print(f"Las gráficas deberían estar en: {plot_output_directory}")
    plt.show() # Muestra todas las figuras al final