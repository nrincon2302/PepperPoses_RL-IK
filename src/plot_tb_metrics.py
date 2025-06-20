import os
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

"""
Script para cargar y graficar métricas de TensorBoard usando matplotlib.
Eje X: pasos * 10
Métricas:
  - curriculum/radius
  - rollout/ep_rew_mean
  - rollout/success_rate
  - train/value_loss
  - train/policy_gradient_loss

Uso:
  python plot_tb_metrics.py --logdir resultados_calibracion/PPO-0/tb_logs --outdir plots
"""

def load_scalars(logdir, tags):
    # Buscar primer archivo de eventos en el directorio
    event_files = [os.path.join(logdir, f) for f in os.listdir(logdir) if f.startswith('events')]
    if not event_files:
        raise FileNotFoundError(f"No se encontraron archivos de eventos en {logdir}")
    ea = EventAccumulator(event_files[0], size_guidance={'scalars': 0})
    ea.Reload()
    data = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            steps = [e.step * 10 for e in events]
            values = [e.value for e in events]
            data[tag] = (steps, values)
        except KeyError:
            print(f"Etiqueta '{tag}' no encontrada en los eventos.")
    return data

def plot_metrics(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    for tag, (steps, values) in data.items():
        plt.figure()
        plt.plot(steps, values)
        plt.xlabel('Timesteps')
        if tag == 'curriculum/radius':
            plt.ylabel('Radio del Curriculum (m)')
            plt.title('Radio del Curriculum a lo largo del tiempo')
        elif tag == 'rollout/ep_rew_mean':
            plt.ylabel('Recompensa Media por Episodio')
            plt.title('Recompensa Media por Episodio a lo largo del tiempo')
        elif tag == 'rollout/success_rate':
            plt.ylabel('Tasa de Éxito')
            plt.title('Tasa de Éxito a lo largo del tiempo')
        elif tag == 'train/value_loss':
            plt.ylabel('Pérdida de Valor')
            plt.title('Pérdida de Valor a lo largo del tiempo')
        elif tag == 'train/policy_gradient_loss':
            plt.ylabel('Pérdida de Gradiente de Política')
            plt.title('Pérdida de Gradiente de Política a lo largo del tiempo')
        else:
            plt.ylabel('Valor')
            plt.title(f'Métrica {tag} a lo largo del tiempo')
        plt.grid(True)
        filename = tag.replace('/', '_') + '.png'
        plt.savefig(os.path.join(outdir, filename))
        plt.close()
        print(f"Guardado: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Graficar métricas de TensorBoard con matplotlib')
    parser.add_argument('--logdir', type=str, required=True, help='Directorio de tb_logs')
    parser.add_argument('--outdir', type=str, default='plots', help='Directorio de salida para las gráficas')
    args = parser.parse_args()

    tags = [
        'curriculum/radius',
        'rollout/ep_rew_mean',
        'rollout/success_rate',
        'train/value_loss',
        'train/policy_gradient_loss',
    ]

    data = load_scalars(args.logdir, tags)
    plot_metrics(data, args.outdir)

if __name__ == '__main__':
    main()