import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

from pepper_env import PepperArmEnv

class PPONetwork(nn.Module):
    """Red neuronal para PPO con arquitectura actor-crítico compartida."""
    
    def __init__(self, state_dim=11, action_dim=5, hidden_dim=256):
        super(PPONetwork, self).__init__()
        
        # Capas compartidas
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: política (media de distribución normal)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Acciones limitadas a [-1, 1], se escalan luego
        )
        
        # Actor: log de desviación estándar
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Crítico: función de valor
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        shared_features = self.shared(state)
        
        # Política
        action_mean = self.actor_mean(shared_features) * 0.05  # Escalar a [-0.05, 0.05]
        action_std = torch.exp(self.actor_logstd)
        
        # Valor
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Obtiene acción de la política."""
        with torch.no_grad():
            action_mean, action_std, value = self.forward(state)
            
            if deterministic:
                return action_mean, value
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action, log_prob, value


class PPOBuffer:
    """Buffer para almacenar experiencias de PPO."""
    
    def __init__(self, size, state_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        
        self.ptr = 0
        self.path_start_idx = 0
        
    def store(self, state, action, reward, value, log_prob, done):
        """Almacena una transición."""
        assert self.ptr < self.size
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        
    def finish_path(self, last_value=0):
        """Calcula returns y ventajas usando GAE al final de un episodio."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Calcular ventajas usando GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)
        
        # Calcular returns
        returns = advantages + self.values[path_slice]
        
        # Almacenar
        self.advantages = getattr(self, 'advantages', np.zeros(self.size, dtype=np.float32))
        self.returns = getattr(self, 'returns', np.zeros(self.size, dtype=np.float32))
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        self.path_start_idx = self.ptr
        
    def get(self):
        """Obtiene todos los datos del buffer."""
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalizar ventajas
        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        return (
            torch.FloatTensor(self.states),
            torch.FloatTensor(self.actions),
            torch.FloatTensor(self.advantages),
            torch.FloatTensor(self.returns),
            torch.FloatTensor(self.log_probs)
        )
    
    def _discount_cumsum(self, x, discount):
        """Calcula suma acumulada descontada."""
        return np.array([np.sum(discount ** np.arange(len(x[i:])) * x[i:]) for i in range(len(x))])


class PPOTrainer:
    """Entrenador PPO."""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Red neuronal
        self.network = PPONetwork(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dim=config['hidden_dim']
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=config['learning_rate'])
        
        # Buffer
        self.buffer = PPOBuffer(
            size=config['steps_per_epoch'],
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda']
        )
        
        # Métricas
        self.train_rewards = []
        self.train_values = []
        self.eval_rewards = []
        self.eval_values = []
        self.episode_lengths = []
        
    def collect_experience(self):
        """Recolecta experiencias usando la política actual."""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_rewards = []
        
        for step in range(self.config['steps_per_epoch']):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.network.get_action(state_tensor)
            
            action_np = action.squeeze(0).numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated
            
            self.buffer.store(
                state, action_np, reward, 
                value.item(), log_prob.item(), done
            )
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                self.buffer.finish_path()
                episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                
        # Finalizar último episodio incompleto
        if episode_length > 0:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, _, value = self.network.get_action(state_tensor)
            self.buffer.finish_path(value.item())
            
        return episode_rewards
    
    def update_policy(self):
        """Actualiza la política usando PPO."""
        states, actions, advantages, returns, old_log_probs = self.buffer.get()
        
        policy_losses = []
        value_losses = []
        
        for _ in range(self.config['train_pi_iters']):
            # Forward pass
            action_means, action_stds, values = self.network(states)
            dist = Normal(action_means, action_stds)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Ratio de importancia
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Loss de política clippado
            clip_ratio = self.config['clip_ratio']
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Loss de valor
            value_loss = ((values.squeeze() - returns) ** 2).mean()
            
            # Loss total
            entropy = dist.entropy().sum(dim=-1).mean()
            total_loss = policy_loss + self.config['vf_coef'] * value_loss - self.config['ent_coef'] * entropy
            
            # Actualización
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            
        return np.mean(policy_losses), np.mean(value_losses)
    
    def evaluate(self, n_episodes=10):
        """Evalúa la política actual."""
        eval_rewards = []
        eval_values = []
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_values = []
            
            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, value = self.network.get_action(state_tensor, deterministic=True)
                
                state, reward, terminated, truncated, _ = self.env.step(action.squeeze(0).numpy())
                episode_reward += reward
                episode_values.append(value.item())
                
                if terminated or truncated:
                    break
                    
            eval_rewards.append(episode_reward)
            eval_values.append(np.mean(episode_values))
            
        return np.mean(eval_rewards), np.mean(eval_values)
    
    def train(self):
        """Bucle principal de entrenamiento."""
        print(f"Iniciando entrenamiento PPO")
        print(f"Configuración: {self.config}")
        
        for epoch in range(self.config['epochs']):
            # Recolectar experiencias
            episode_rewards = self.collect_experience()
            
            # Actualizar política
            policy_loss, value_loss = self.update_policy()
            
            # Evaluación periódica
            if epoch % self.config['eval_frequency'] == 0:
                eval_reward, eval_value = self.evaluate()
                self.eval_rewards.append(eval_reward)
                self.eval_values.append(eval_value)
                
                print(f"Época {epoch:4d} | "
                      f"Train R: {np.mean(episode_rewards):7.2f} | "
                      f"Eval R: {eval_reward:7.2f} | "
                      f"Eval V: {eval_value:7.2f} | "
                      f"Policy Loss: {policy_loss:.4f} | "
                      f"Value Loss: {value_loss:.4f}")
            
            # Guardar métricas de entrenamiento
            if episode_rewards:
                self.train_rewards.append(np.mean(episode_rewards))
                # Calcular valor promedio durante entrenamiento
                states, _, _, returns, _ = self.buffer.get()
                train_value = returns.mean().item()
                self.train_values.append(train_value)
            
            # Guardar modelo periódicamente
            if epoch % self.config['save_frequency'] == 0:
                self.save_model(f"models/ppo_pepper_epoch_{epoch}.pth")
        
        # Guardar modelo final
        self.save_model("models/ppo_pepper_final.pth")
        self.save_metrics()
        
    def save_model(self, path):
        """Guarda el modelo."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
    def save_metrics(self):
        """Guarda las métricas de entrenamiento."""
        metrics = {
            'train_rewards': self.train_rewards,
            'train_values': self.train_values,
            'eval_rewards': self.eval_rewards,
            'eval_values': self.eval_values,
            'episode_lengths': self.episode_lengths,
            'config': self.config
        }
        
        os.makedirs("results", exist_ok=True)
        with open("results/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


def main():
    # Configuración de entrenamiento
    config = {
        'epochs': 1000,
        'steps_per_epoch': 4000,
        'hidden_dim': 256,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'train_pi_iters': 80,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
        'eval_frequency': 10,
        'save_frequency': 100
    }
    
    # Crear entorno
    env = PepperArmEnv(
        side='Left',
        render_mode=None,
        max_steps=250,
        n_workspace_samples=8,
        curriculum_start_frac=0.2,
        curriculum_increment_frac=0.1
    )
    
    # Entrenar
    trainer = PPOTrainer(env, config)
    trainer.train()
    
    print("Entrenamiento completado!")


if __name__ == "__main__":
    main()