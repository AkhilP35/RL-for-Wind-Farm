"""
PPO (Proximal Policy Optimization) Agent
PyTorch implementation for wind farm control
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import logging

import config

logger = logging.getLogger(__name__)


class Actor(nn.Module):
    """Actor network for policy (yaw angle control)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int], activation: str = "relu"):
        super(Actor, self).__init__()
        
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = hidden_size
        
        self.network = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)
        
        # Initialize output layers
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        self.mean_layer.bias.data.zero_()
        self.log_std_layer.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor, shape (batch_size, state_dim)
            
        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Clamp log_std for stability
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clip action to valid range
        action = torch.clamp(action, config.YAW_MIN, config.YAW_MAX)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim: int, hidden_sizes: List[int], activation: str = "relu"):
        super(Critic, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = hidden_size
        
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor, shape (batch_size, state_dim)
            
        Returns:
            Value estimate, shape (batch_size, 1)
        """
        return self.network(state)


class PPOAgent:
    """
    PPO Agent for wind farm control
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize PPO Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: PyTorch device (CPU/GPU)
            **kwargs: Additional configuration (overrides config.py)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters (from config or kwargs)
        self.lr = kwargs.get('learning_rate', config.PPO_CONFIG['learning_rate'])
        self.gamma = kwargs.get('gamma', config.PPO_CONFIG['gamma'])
        self.lambda_gae = kwargs.get('lambda', config.PPO_CONFIG['lambda'])
        self.clip_epsilon = kwargs.get('clip_epsilon', config.PPO_CONFIG['clip_epsilon'])
        self.entropy_coef = kwargs.get('entropy_coef', config.PPO_CONFIG['entropy_coef'])
        self.value_coef = kwargs.get('value_coef', config.PPO_CONFIG['value_coef'])
        self.max_grad_norm = kwargs.get('max_grad_norm', config.PPO_CONFIG['max_grad_norm'])
        self.update_epochs = kwargs.get('update_epochs', config.PPO_CONFIG['update_epochs'])
        self.batch_size = kwargs.get('batch_size', config.PPO_CONFIG['batch_size'])
        
        # Network architecture
        actor_hidden = kwargs.get('actor_hidden_sizes', config.NETWORK_CONFIG['actor_hidden_sizes'])
        critic_hidden = kwargs.get('critic_hidden_sizes', config.NETWORK_CONFIG['critic_hidden_sizes'])
        activation = kwargs.get('activation', config.NETWORK_CONFIG['activation'])
        
        # Create networks
        self.actor = Actor(state_dim, action_dim, actor_hidden, activation).to(self.device)
        self.critic = Critic(state_dim, critic_hidden, activation).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        logger.info(f"PPO Agent initialized: state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action from policy
        
        Args:
            state: State array
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
        
        action = action.cpu().numpy().flatten()
        log_prob = log_prob.cpu().item() if log_prob is not None else 0.0
        value = value.cpu().item()
        
        return action, log_prob, value
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states
        
        Args:
            states: State tensors
            actions: Action tensors
            
        Returns:
            Tuple of (log_probs, values)
        """
        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        values = self.critic(states)
        
        return log_probs, values
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * values[step + 1] - values[step]
                gae = delta + self.gamma * self.lambda_gae * gae
            
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values[:-1])
        
        return advantages, returns
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Update policy and value networks using PPO
        
        Args:
            states: State arrays
            actions: Action arrays
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            
        Returns:
            Dictionary of training statistics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Training statistics
        stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
        }
        
        # Update for multiple epochs
        for epoch in range(self.update_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            # Mini-batch updates
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values = self.evaluate_actions(batch_states, batch_actions)
                entropy = Normal(
                    *self.actor(batch_states)
                ).entropy().sum(dim=-1).mean()
                
                # Compute ratios
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Actor loss (PPO clipped)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Critic loss
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    stats['actor_loss'] += actor_loss.item()
                    stats['critic_loss'] += critic_loss.item()
                    stats['entropy'] += entropy.item()
                    stats['approx_kl'] += approx_kl
        
        # Average statistics
        n_updates = self.update_epochs * (len(states) // self.batch_size + 1)
        for key in stats:
            stats[key] /= n_updates
        
        return stats
    
    def save(self, filepath: str):
        """Save agent to file"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        logger.info(f"Agent loaded from {filepath}")
