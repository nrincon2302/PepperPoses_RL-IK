# scripts/train_ppo_5dof.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import pandas as pd
import time

from envs.pepper_arm_env import PepperArmEnv5DOF # Asegúrate que el path sea correcto
from curriculum.scheduler import CurriculumScheduler # Asegúrate que el path sea correcto

class CurriculumAndStatsCallback(BaseCallback):
    """
    Callback para gestionar el avance del currículo y registrar estadísticas detalladas.
    Avanza el currículo cada `advance_curriculum_every_n_timesteps`.
    Registra estadísticas por episodio y las guarda al final del entrenamiento.
    """
    def __init__(self,
                 env_unwrapped, # Pasar el entorno sin wrappers como Monitor
                 advance_curriculum_every_n_timesteps: int,
                 log_path: str,
                 num_curriculum_levels: int,
                 verbose=0):
        super().__init__(verbose)
        self.env_unwrapped = env_unwrapped # Necesario para acceder a scheduler.advance()
        self.advance_curriculum_every_n_timesteps = advance_curriculum_every_n_timesteps
        self.log_path = log_path
        self.num_curriculum_levels = num_curriculum_levels
        
        # Para estadísticas por episodio
        self.episode_stats = []
        self.current_episode_rewards = []
        self.current_episode_curriculum_level = env_unwrapped.scheduler.level

    def _on_training_start(self) -> None:
        # Crear directorio de logs si no existe
        os.makedirs(self.log_path, exist_ok=True)
        # Registrar nivel inicial del currículo
        self.logger.record("curriculum/level", self.env_unwrapped.scheduler.level)


    def _on_step(self) -> bool:
        # Comprobar si es momento de avanzar el currículo
        # self.num_timesteps es el número total de timesteps de la simulación hasta ahora
        if self.num_timesteps % self.advance_curriculum_every_n_timesteps == 0 and self.num_timesteps > 0:
            if self.env_unwrapped.scheduler.level < self.num_curriculum_levels -1:
                self.env_unwrapped.scheduler.advance()
                if self.verbose > 0:
                    print(f"Callback: Curriculum advanced to level {self.env_unwrapped.scheduler.level} at timestep {self.num_timesteps}")
                # Registrar el nuevo nivel del currículo en TensorBoard
                self.logger.record("curriculum/level", self.env_unwrapped.scheduler.level)

        # Recopilar datos del episodio (funciona si env está envuelto en Monitor)
        # Monitor guarda 'r' (reward), 'l' (length), 't' (time)
        # Necesitamos acceder a info para 'distance' y 'is_success'
        # Esto se hará cuando Monitor detecte el final de un episodio.
        # Sin embargo, Monitor no guarda el nivel del currículo.
        # Así que capturaremos la info al final de cada episodio.

        # 'dones' es un array, incluso para un solo entorno
        if 'dones' in self.locals and self.locals['dones'][0]:
            info = self.locals['infos'][0] # Info del entorno
            
            # 'episode' key es añadida por el wrapper Monitor
            if 'episode' in info:
                ep_rew = info['episode']['r']
                ep_len = info['episode']['l']
                
                # Datos adicionales de nuestro entorno
                ep_dist = info.get('distance', np.nan)
                ep_success = info.get('is_success', False) # is_success debe ser True si terminated
                
                self.episode_stats.append({
                    'timestep': self.num_timesteps,
                    'curriculum_level': self.current_episode_curriculum_level, # Nivel al inicio del episodio
                    'reward': ep_rew,
                    'length': ep_len,
                    'final_distance': ep_dist,
                    'success': ep_success
                })
            # Actualizar el nivel de currículo para el *próximo* episodio, si es que cambió
            self.current_episode_curriculum_level = self.env_unwrapped.scheduler.level
        return True

    def _on_training_end(self) -> None:
        if not self.episode_stats:
            if self.verbose > 0:
                print("Callback: No episode stats collected to save.")
            return

        df_episode_stats = pd.DataFrame(self.episode_stats)
        stats_filename = os.path.join(self.log_path, 'detailed_curriculum_episode_stats.csv')
        df_episode_stats.to_csv(stats_filename, index=False)
        if self.verbose > 0:
            print(f"Callback: Detailed episode curriculum stats saved to {stats_filename}")

        # Calcular estadísticas agregadas por nivel de currículo
        if not df_episode_stats.empty:
            agg_stats = df_episode_stats.groupby('curriculum_level').agg(
                mean_reward=('reward', 'mean'),
                std_reward=('reward', 'std'),
                mean_length=('length', 'mean'),
                success_rate=('success', 'mean'), # 'mean' de booleanos (0/1) es la tasa
                mean_final_distance=('final_distance', 'mean'),
                episodes_in_level=('reward', 'count')
            ).reset_index()
            
            agg_filename = os.path.join(self.log_path, 'aggregated_curriculum_stats.csv')
            agg_stats.to_csv(agg_filename, index=False)
            if self.verbose > 0:
                print(f"Callback: Aggregated curriculum stats saved to {agg_filename}")


def train(output_dir: str = "training_results", 
          total_timesteps: int = 1_000_000, 
          curriculum_levels: int = 10,
          advance_curriculum_every_n_timesteps: int = 100_000,
          env_max_steps: int = 250,
          history_size: int = 3
          ):
    
    run_name = f"ppo_pepper5dof_{int(time.time())}"
    current_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(current_output_dir, exist_ok=True)
    
    monitor_log_dir = os.path.join(current_output_dir, "monitor_logs")
    os.makedirs(monitor_log_dir, exist_ok=True)

    # Crear el entorno
    # Necesitamos la instancia no envuelta para el callback del currículo
    env_unwrapped = PepperArmEnv5DOF(render_mode=None, max_steps=env_max_steps, history_size=history_size)
    env_unwrapped.scheduler = CurriculumScheduler(num_curriculum_levels=curriculum_levels) # Sobrescribir scheduler
    
    # Envolver el entorno con Monitor para logging estándar de SB3
    env = Monitor(env_unwrapped, monitor_log_dir)

    # Callback personalizado
    # Pasar env_unwrapped para que el callback pueda acceder a .scheduler
    curriculum_callback = CurriculumAndStatsCallback(
        env_unwrapped=env_unwrapped,
        advance_curriculum_every_n_timesteps=advance_curriculum_every_n_timesteps,
        log_path=current_output_dir, # Guardar CSVs en el directorio del run
        num_curriculum_levels=curriculum_levels,
        verbose=1
    )
    
    # Modelo PPO
    # El tensorboard_log se guardará en current_output_dir/ppo_pepper_tensorboard_RUN_NAME
    model = PPO('MlpPolicy',
                env,
                verbose=1,
                tensorboard_log=os.path.join(current_output_dir, "tensorboard_logs"),
                n_steps=2048, # Ajustar según sea necesario
                batch_size=64, # Ajustar según sea necesario
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01, # Ajustar entropía
                vf_coef=0.5,
                # learning_rate=3e-4, # Puedes ajustar esto también
                )
    
    print(f"Starting training. Total timesteps: {total_timesteps}")
    print(f"Max episode steps: {env_max_steps}")
    print(f"Curriculum levels: {curriculum_levels}")
    print(f"Advance curriculum every: {advance_curriculum_every_n_timesteps} timesteps")
    print(f"Output directory: {current_output_dir}")
    print(f"Tensorboard log directory: {os.path.join(current_output_dir, 'tensorboard_logs')}")

    model.learn(total_timesteps=int(total_timesteps),
                callback=curriculum_callback,
                progress_bar=True,
                tb_log_name=run_name # Nombre para la subcarpeta de TensorBoard
                )
    
    model_save_path = os.path.join(current_output_dir, 'ppo_pepper5dof_final.zip')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Monitor logs saved in: {monitor_log_dir}")
    print(f"Curriculum stats CSVs saved in: {current_output_dir}")
    print(f"To view TensorBoard logs: tensorboard --logdir {os.path.join(current_output_dir, 'tensorboard_logs')}")

if __name__ == '__main__':
    train(
        output_dir="training_results_pepper",
        total_timesteps=500_000, # Reducido para prueba rápida, aumenta para entrenamiento real
        curriculum_levels=5,
        advance_curriculum_every_n_timesteps=100_000, # Avanza currículo cada 100k pasos
        env_max_steps=200, # Pasos por episodio
        history_size=3
    )