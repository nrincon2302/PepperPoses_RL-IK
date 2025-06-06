import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pepper_env import PepperArmEnv
from train_pepper import PPONetwork
import argparse
import os

class PPOEvaluator:
    """Evaluador para modelos PPO entrenados."""
    
    def __init__(self, model_path, env_config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar modelo
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Crear red neuronal
        self.network = PPONetwork(
            state_dim=11,  # Observación del PepperArmEnv
            action_dim=5,  # Acciones del PepperArmEnv
            hidden_dim=self.config['hidden_dim']
        )
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()
        
        # Crear entorno
        env_config = env_config or {}
        self.env = PepperArmEnv(
            side=env_config.get('side', 'Left'),
            render_mode=env_config.get('render_mode', None),
            max_steps=env_config.get('max_steps', 250),
            n_workspace_samples=env_config.get('n_workspace_samples', 8),
            curriculum_start_frac=env_config.get('curriculum_start_frac', 1.0),  # Evaluar en todo el workspace
            curriculum_increment_frac=env_config.get('curriculum_increment_frac', 0.0)
        )
        
    def evaluate_episodes(self, n_episodes=100, render=False, verbose=True, track_joints=False):
        """Evalúa el modelo en múltiples episodios."""
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': 0,
            'final_distances': [],
            'episode_values': [],
            'joint_trajectories': [],  # Nuevo: trayectorias de ángulos
            'position_trajectories': [],  # Nuevo: trayectorias de posición
            'reward_trajectories': [],  # Nuevo: trayectorias de recompensa
            'value_trajectories': []   # Nuevo: trayectorias de valor
        }
        
        successes = 0
        
        for episode in range(n_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_values = []
            
            # Tracking para este episodio
            if track_joints or episode < 5:  # Trackear los primeros 5 episodios por defecto
                joint_history = []
                position_history = []
                reward_history = []
                value_history = []
            
            while True:
                # Obtener acción determinística
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, value = self.network.get_action(state_tensor, deterministic=True)
                
                # Trackear datos si es necesario
                if track_joints or episode < 5:
                    joint_history.append(self.env.joint_angles.copy())
                    position_history.append(self.env.current_pos.copy())
                    value_history.append(value.item())
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, info = self.env.step(action.squeeze(0).numpy())
                
                episode_reward += reward
                episode_length += 1
                episode_values.append(value.item())
                
                # Trackear recompensa
                if track_joints or episode < 5:
                    reward_history.append(reward)
                
                if render:
                    self.env.render()
                
                if terminated or truncated:
                    if terminated:  # Éxito
                        successes += 1
                    results['final_distances'].append(info['distance'])
                    
                    # Guardar trayectorias
                    if track_joints or episode < 5:
                        results['joint_trajectories'].append(np.array(joint_history))
                        results['position_trajectories'].append(np.array(position_history))
                        results['reward_trajectories'].append(np.array(reward_history))
                        results['value_trajectories'].append(np.array(value_history))
                    
                    break
                    
                state = next_state
            
            results['episode_rewards'].append(episode_reward)
            results['episode_lengths'].append(episode_length)
            results['episode_values'].append(np.mean(episode_values))
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episodio {episode + 1:3d} | "
                      f"Recompensa: {episode_reward:7.2f} | "
                      f"Longitud: {episode_length:3d} | "
                      f"Distancia final: {info['distance']:.4f}")
        
        results['success_rate'] = successes / n_episodes
        
        if verbose:
            self._print_summary(results)
            
        return results
    
    def evaluate_single_episode_detailed(self, target_pos=None, render=False):
        """Evalúa un solo episodio con seguimiento detallado de todas las variables."""
        if target_pos is not None:
            state, info = self.env.reset(options={'target_pos': target_pos})
        else:
            state, info = self.env.reset()
        
        # Almacenar datos detallados
        detailed_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'joint_angles': [],
            'end_effector_positions': [],
            'distances_to_target': [],
            'target_position': self.env.target_pos.copy(),
            'initial_distance': info['distance']
        }
        
        episode_reward = 0
        step = 0
        
        while True:
            # Guardar estado actual
            detailed_data['states'].append(state.copy())
            detailed_data['joint_angles'].append(self.env.joint_angles.copy())
            detailed_data['end_effector_positions'].append(self.env.current_pos.copy())
            detailed_data['distances_to_target'].append(
                np.linalg.norm(self.env.target_pos - self.env.current_pos)
            )
            
            # Obtener acción y valor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, value = self.network.get_action(state_tensor, deterministic=True)
            
            action_numpy = action.squeeze(0).numpy()
            detailed_data['actions'].append(action_numpy.copy())
            detailed_data['values'].append(value.item())
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, info = self.env.step(action_numpy)
            
            detailed_data['rewards'].append(reward)
            episode_reward += reward
            step += 1
            
            if render:
                self.env.render()
            
            if terminated or truncated:
                # Añadir estado final
                detailed_data['states'].append(next_state.copy())
                detailed_data['joint_angles'].append(self.env.joint_angles.copy())
                detailed_data['end_effector_positions'].append(self.env.current_pos.copy())
                detailed_data['distances_to_target'].append(info['distance'])
                
                detailed_data['success'] = terminated
                detailed_data['episode_length'] = step
                detailed_data['episode_reward'] = episode_reward
                detailed_data['final_distance'] = info['distance']
                break
                
            state = next_state
        
        # Convertir listas a arrays numpy
        for key in ['states', 'actions', 'rewards', 'values', 'joint_angles', 
                   'end_effector_positions', 'distances_to_target']:
            detailed_data[key] = np.array(detailed_data[key])
        
        return detailed_data
    
    def _print_summary(self, results):
        """Imprime resumen de evaluación."""
        print("\n" + "="*60)
        print("RESUMEN DE EVALUACIÓN")
        print("="*60)
        print(f"Recompensa promedio:     {np.mean(results['episode_rewards']):7.2f} ± {np.std(results['episode_rewards']):6.2f}")
        print(f"Longitud promedio:       {np.mean(results['episode_lengths']):7.1f} ± {np.std(results['episode_lengths']):6.1f}")
        print(f"Tasa de éxito:          {results['success_rate']*100:7.1f}%")
        print(f"Distancia final promedio: {np.mean(results['final_distances']):6.4f} ± {np.std(results['final_distances']):6.4f}")
        print(f"Valor promedio:         {np.mean(results['episode_values']):7.2f} ± {np.std(results['episode_values']):6.2f}")
        print("="*60)
    
    def evaluate_curriculum_progression(self, n_episodes_per_level=20):
        """Evalúa el rendimiento en diferentes niveles de dificultad del curriculum."""
        curriculum_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        results = []
        
        for level in curriculum_levels:
            print(f"\nEvaluando nivel de curriculum: {level:.1f}")
            
            # Configurar entorno para este nivel
            self.env.curriculum_radius = level * self.env.max_distance
            
            level_results = self.evaluate_episodes(
                n_episodes=n_episodes_per_level, 
                render=False, 
                verbose=False
            )
            
            level_results['curriculum_level'] = level
            results.append(level_results)
            
            print(f"Nivel {level:.1f} - Recompensa: {np.mean(level_results['episode_rewards']):.2f}, "
                  f"Éxito: {level_results['success_rate']*100:.1f}%")
        
        return results
    
    def evaluate_specific_targets(self, target_positions, render=False, detailed=False):
        """Evalúa el modelo en posiciones objetivo específicas."""
        results = []
        
        for i, target_pos in enumerate(target_positions):
            print(f"\nEvaluando objetivo {i+1}: {target_pos}")
            
            if detailed:
                result = self.evaluate_single_episode_detailed(target_pos, render)
                result['target_index'] = i
            else:
                state, _ = self.env.reset(options={'target_pos': target_pos})
                episode_reward = 0
                episode_length = 0
                
                while True:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action, _ = self.network.get_action(state_tensor, deterministic=True)
                    
                    next_state, reward, terminated, truncated, info = self.env.step(action.squeeze(0).numpy())
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if render:
                        self.env.render()
                    
                    if terminated or truncated:
                        result = {
                            'target_position': target_pos,
                            'episode_reward': episode_reward,
                            'episode_length': episode_length,
                            'success': terminated,
                            'final_distance': info['distance'],
                            'target_index': i
                        }
                        break
                        
                    state = next_state
            
            results.append(result)
            
            success_text = 'Sí' if result.get('success', False) else 'No'
            final_dist = result.get('final_distance', 0)
            reward = result.get('episode_reward', 0)
            
            print(f"Objetivo {i+1} - Recompensa: {reward:.2f}, "
                  f"Éxito: {success_text}, "
                  f"Distancia: {final_dist:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo PPO entrenado')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Ruta al modelo entrenado')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Número de episodios para evaluación')
    parser.add_argument('--render', action='store_true',
                        help='Renderizar episodios')
    parser.add_argument('--curriculum', action='store_true',
                        help='Evaluar progresión del curriculum')
    parser.add_argument('--side', type=str, default='Left', choices=['Left', 'Right'],
                        help='Brazo a evaluar')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Ruta para guardar resultados JSON')
    parser.add_argument('--track_joints', action='store_true',
                        help='Trackear evolución de ángulos de articulaciones')
    parser.add_argument('--detailed_analysis', action='store_true',
                        help='Realizar análisis detallado de episodios específicos')
    
    args = parser.parse_args()
    
    # Crear evaluador
    env_config = {'side': args.side}
    if args.render:
        env_config['render_mode'] = 'human'
    
    evaluator = PPOEvaluator(args.model_path, env_config)
    
    # Evaluación estándar
    print(f"Evaluando modelo: {args.model_path}")
    print(f"Número de episodios: {args.n_episodes}")
    
    results = evaluator.evaluate_episodes(
        n_episodes=args.n_episodes,
        render=args.render,
        track_joints=args.track_joints
    )
    
    # Evaluación de curriculum si se solicita
    if args.curriculum:
        curriculum_results = evaluator.evaluate_curriculum_progression()
        results['curriculum_evaluation'] = curriculum_results
    
    # Evaluación en objetivos específicos (esquinas del workspace)
    print("\nEvaluando objetivos específicos...")
    specific_targets = [
        [0.3, 0.2, 0.8],   # Frente derecha
        [0.3, -0.2, 0.8],  # Frente izquierda  
        [0.1, 0.3, 1.0],   # Arriba derecha
        [0.1, -0.3, 1.0],  # Arriba izquierda
        [0.4, 0.0, 0.6]    # Centro frente
    ]
    
    specific_results = evaluator.evaluate_specific_targets(
        specific_targets, 
        render=args.render,
        detailed=args.detailed_analysis
    )
    results['specific_targets'] = specific_results
    
    # Análisis detallado si se solicita
    if args.detailed_analysis:
        print("\nRealizando análisis detallado...")
        detailed_episode = evaluator.evaluate_single_episode_detailed(
            target_pos=specific_targets[0],
            render=args.render
        )
        results['detailed_episode'] = detailed_episode
    
    # Guardar resultados si se especifica
    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        
        # Convertir arrays numpy a listas para JSON
        def convert_numpy_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_arrays(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_arrays(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy_arrays(results)
        
        with open(args.save_results, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        print(f"\nResultados guardados en: {args.save_results}")


if __name__ == "__main__":
    main()