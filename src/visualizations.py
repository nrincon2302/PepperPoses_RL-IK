import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from datetime import datetime

# Configurar estilo de matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PepperVisualizationTool:
    """Herramienta de visualización para análisis de entrenamiento y evaluación de Pepper."""
    
    def __init__(self, results_path=None, training_metrics_path=None):
        """
        Inicializa la herramienta de visualización.
        
        Args:
            results_path: Ruta al archivo JSON con resultados de evaluación
            training_metrics_path: Ruta al archivo JSON con métricas de entrenamiento
        """
        self.results = None
        self.training_metrics = None
        
        if results_path and os.path.exists(results_path):
            with open(results_path, 'r') as f:
                self.results = json.load(f)
        
        if training_metrics_path and os.path.exists(training_metrics_path):
            with open(training_metrics_path, 'r') as f:
                self.training_metrics = json.load(f)
    
    def plot_training_evolution(self, save_path=None):
        """Gráfica de evolución durante el entrenamiento."""
        if not self.training_metrics:
            print("No hay datos de entrenamiento disponibles.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evolución del Entrenamiento PPO - Pepper Arm', fontsize=16, fontweight='bold')
        
        # Recompensas de entrenamiento
        axes[0, 0].plot(self.training_metrics['train_rewards'], 'b-', linewidth=2, alpha=0.7, label='Entrenamiento')
        if self.training_metrics['eval_rewards']:
            eval_epochs = np.arange(0, len(self.training_metrics['train_rewards']), 
                                  len(self.training_metrics['train_rewards']) // len(self.training_metrics['eval_rewards']))[:len(self.training_metrics['eval_rewards'])]
            axes[0, 0].plot(eval_epochs, self.training_metrics['eval_rewards'], 'r-', 
                          linewidth=2, marker='o', markersize=4, label='Evaluación')
        axes[0, 0].set_title('Recompensas por Época')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Recompensa')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Función de valor
        axes[0, 1].plot(self.training_metrics['train_values'], 'g-', linewidth=2, alpha=0.7, label='Entrenamiento')
        if self.training_metrics['eval_values']:
            axes[0, 1].plot(eval_epochs, self.training_metrics['eval_values'], 'orange', 
                          linewidth=2, marker='s', markersize=4, label='Evaluación')
        axes[0, 1].set_title('Función de Valor por Época')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Valor Promedio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Longitud de episodios
        if self.training_metrics['episode_lengths']:
            # Calcular promedio móvil para suavizar
            window_size = max(1, len(self.training_metrics['episode_lengths']) // 50)
            smoothed_lengths = np.convolve(self.training_metrics['episode_lengths'], 
                                         np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(self.training_metrics['episode_lengths'], 'lightblue', alpha=0.5, label='Episodios individuales')
            axes[1, 0].plot(smoothed_lengths, 'darkblue', linewidth=2, label=f'Promedio móvil (ventana={window_size})')
        else:
            axes[1, 0].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        axes[1, 0].set_title('Longitud de Episodios')
        axes[1, 0].set_xlabel('Episodio')
        axes[1, 0].set_ylabel('Pasos')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribución de recompensas finales
        axes[1, 1].hist(self.training_metrics['train_rewards'], bins=30, alpha=0.7, 
                       color='blue', edgecolor='black', label='Entrenamiento')
        if self.training_metrics['eval_rewards']:
            axes[1, 1].hist(self.training_metrics['eval_rewards'], bins=15, alpha=0.7, 
                           color='red', edgecolor='black', label='Evaluación')
        axes[1, 1].set_title('Distribución de Recompensas')
        axes[1, 1].set_xlabel('Recompensa')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica de entrenamiento guardada en: {save_path}")
        
        plt.show()
    
    def plot_joint_evolution(self, episode_idx=0, save_path=None):
        """Gráfica de evolución de ángulos de articulaciones durante un episodio."""
        if not self.results or 'joint_trajectories' not in self.results:
            print("No hay datos de trayectorias de articulaciones disponibles.")
            return
        
        if episode_idx >= len(self.results['joint_trajectories']):
            print(f"Índice de episodio {episode_idx} no válido. Máximo: {len(self.results['joint_trajectories'])-1}")
            return
        
        joint_trajectory = np.array(self.results['joint_trajectories'][episode_idx])
        
        # Nombres de las articulaciones del brazo izquierdo de Pepper
        joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Evolución de Ángulos de Articulaciones - Episodio {episode_idx}', 
                    fontsize=16, fontweight='bold')
        
        # Plotear cada articulación
        for i, joint_name in enumerate(joint_names):
            row = i // 2
            col = i % 2
            
            angles_deg = np.degrees(joint_trajectory[:, i])  # Convertir a grados
            steps = np.arange(len(angles_deg))
            
            axes[row, col].plot(steps, angles_deg, linewidth=2, marker='o', markersize=3)
            axes[row, col].set_title(f'{joint_name}')
            axes[row, col].set_xlabel('Paso')
            axes[row, col].set_ylabel('Ángulo (grados)')
            axes[row, col].grid(True, alpha=0.3)
            
            # Añadir líneas de referencia para límites si están disponibles
            if hasattr(self, 'joint_limits'):
                axes[row, col].axhline(y=np.degrees(self.joint_limits[joint_name][0]), 
                                     color='red', linestyle='--', alpha=0.5, label='Límite inferior')
                axes[row, col].axhline(y=np.degrees(self.joint_limits[joint_name][1]), 
                                     color='red', linestyle='--', alpha=0.5, label='Límite superior')
        
        # Gráfica adicional: Todas las articulaciones juntas
        axes[2, 1].clear()
        for i, joint_name in enumerate(joint_names):
            angles_deg = np.degrees(joint_trajectory[:, i])
            axes[2, 1].plot(steps, angles_deg, linewidth=2, label=joint_name, marker='o', markersize=2)
        
        axes[2, 1].set_title('Todas las Articulaciones')
        axes[2, 1].set_xlabel('Paso')
        axes[2, 1].set_ylabel('Ángulo (grados)')
        axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Información del episodio
        if 'episode_rewards' in self.results and episode_idx < len(self.results['episode_rewards']):
            episode_reward = self.results['episode_rewards'][episode_idx]
            episode_length = len(joint_trajectory)
            success = episode_idx < len(self.results['final_distances']) and self.results['final_distances'][episode_idx] <= 0.02
            
            info_text = f"Recompensa: {episode_reward:.2f}\nLongitud: {episode_length} pasos\nÉxito: {'Sí' if success else 'No'}"
            axes[2, 0].text(0.1, 0.7, info_text, transform=axes[2, 0].transAxes, 
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        axes[2, 0].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica de evolución de articulaciones guardada en: {save_path}")
        
        plt.show()
    
    def plot_position_trajectory_3d(self, episode_idx=0, save_path=None):
        """Gráfica 3D de la trayectoria del efector final."""
        if not self.results or 'position_trajectories' not in self.results:
            print("No hay datos de trayectorias de posición disponibles.")
            return
        
        if episode_idx >= len(self.results['position_trajectories']):
            print(f"Índice de episodio {episode_idx} no válido.")
            return
        
        position_trajectory = np.array(self.results['position_trajectories'][episode_idx])
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotear trayectoria
        ax.plot(position_trajectory[:, 0], position_trajectory[:, 1], position_trajectory[:, 2], 
               'b-', linewidth=2, alpha=0.7, label='Trayectoria')
        
        # Marcar posición inicial y final
        ax.scatter(position_trajectory[0, 0], position_trajectory[0, 1], position_trajectory[0, 2], 
                  color='green', s=100, marker='o', label='Inicio')
        ax.scatter(position_trajectory[-1, 0], position_trajectory[-1, 1], position_trajectory[-1, 2], 
                  color='red', s=100, marker='s', label='Final')
        
        # Marcar objetivo si está disponible
        if 'specific_targets' in self.results and len(self.results['specific_targets']) > episode_idx:
            target = self.results['specific_targets'][episode_idx]['target_position']
            ax.scatter(target[0], target[1], target[2], 
                      color='gold', s=150, marker='*', label='Objetivo')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Trayectoria 3D del Efector Final - Episodio {episode_idx}', fontweight='bold')
        ax.legend()
        
        # Añadir información del episodio
        if 'episode_rewards' in self.results and episode_idx < len(self.results['episode_rewards']):
            episode_reward = self.results['episode_rewards'][episode_idx]
            final_distance = self.results['final_distances'][episode_idx] if episode_idx < len(self.results['final_distances']) else 'N/A'
            success = 'Sí' if final_distance <= 0.02 else 'No'
            info_text = f"Recompensa: {episode_reward:.2f}\nDistancia Final: {final_distance:.3f} m\nÉxito: {success}"
            ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica de trayectoria 3D guardada en: {save_path}")
        plt.show()

        