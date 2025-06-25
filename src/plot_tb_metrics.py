import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Configuración de directorios
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SIDES = ["Left", "Right"]
ALGOS = ["PPO", "SAC"]
RUNS = range(4)  # 0,1,2,3

# Métricas a graficar: tag en TensorBoard y título
METRICS = [
    ("curriculum/radius", "Radio de Currículo", "curriculum_radius.png"),
    ("rollout/ep_rew_mean", "Recompensa Media en entrenamiento", "ep_rew_mean.png"),
    ("rollout/success_rate", "Tasa de Éxito Promedio en Ventana de 100 episodios", "success_rate.png")
]

# Carpeta de salida
GRAPHS_DIR = os.path.join(SRC_DIR, "graphs")

if __name__ == "__main__":
    for algo in ALGOS:
        for side in SIDES:
            # Crear carpetas de salida
            out_base = os.path.join(GRAPHS_DIR, algo, side)
            os.makedirs(out_base, exist_ok=True)

            # Recolectar datos para cada corrida
            data = {}
            for run in RUNS:
                run_name = f"{algo}-analytical-{run}"
                log_dir = os.path.join(SRC_DIR, f"resultados_{side}", run_name, "tb_logs")
                if not os.path.isdir(log_dir):
                    print(f"Directorio no encontrado: {log_dir}")
                    continue
                ea = event_accumulator.EventAccumulator(log_dir)
                ea.Reload()
                # Para cada métrica, extraer pasos y valores
                for tag, _, _ in METRICS:
                    if tag not in data:
                        data[tag] = {}
                    try:
                        events = ea.Scalars(tag)
                        steps = [e.step for e in events]
                        vals  = [e.value for e in events]
                        data[tag][run_name] = (steps, vals)
                    except KeyError:
                        print(f"Tag no encontrado {tag} en {run_name}")

            # Generar gráficas comparativas
            for tag, title, filename in METRICS:
                plt.figure()
                for run_name, (steps, vals) in sorted(data[tag].items()):
                    plt.plot(steps, vals, label=run_name)
                plt.title(title)
                plt.xlabel("Paso")
                plt.ylabel(title)
                plt.legend()
                plt.grid(True)
                out_path = os.path.join(out_base, filename)
                plt.savefig(out_path)
                plt.close()
                print(f"Guardado: {out_path}")
