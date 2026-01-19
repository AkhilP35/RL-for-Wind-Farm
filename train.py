"""
Training script for PPO Wind Farm Control
"""

import numpy as np
import torch
import argparse
import os
from pathlib import Path
from typing import List, Dict
import logging

import config
from wind_farm_env import WindFarmEnv
from ppo_agent import PPOAgent
import utils

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Buffer for storing experience during episode collection"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, reward, log_prob, value, done):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def get_arrays(self):
        """Get all data as numpy arrays"""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'dones': np.array(self.dones),
        }


def train_episode(env: WindFarmEnv, agent: PPOAgent, buffer: ExperienceBuffer) -> Dict:
    """
    Collect experience for one episode
    
    Returns:
        Dictionary with episode statistics
    """
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_powers = []
    
    done = False
    while not done:
        # Select action
        action, log_prob, value = agent.select_action(obs)
        
        # Step environment
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        
        # Store experience
        buffer.add(obs, action, reward, log_prob, value, done)
        
        # Update
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        episode_powers.append(step_info['power'])
    
    # Get final value estimate
    _, _, final_value = agent.select_action(obs)
    buffer.values.append(final_value)
    
    return {
        'reward': episode_reward,
        'length': episode_length,
        'powers': episode_powers,
        'final_power': episode_powers[-1] if episode_powers else 0.0,
        'baseline_power': info.get('baseline_power', 0.0),
    }


def train(
    env: WindFarmEnv,
    agent: PPOAgent,
    total_episodes: int,
    eval_frequency: int = 50,
    eval_episodes: int = 10,
    save_frequency: int = 100,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
):
    """
    Main training loop
    
    Args:
        env: Wind farm environment
        agent: PPO agent
        total_episodes: Total number of training episodes
        eval_frequency: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
        save_frequency: Save checkpoint every N episodes
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    training_stats = []
    
    buffer = ExperienceBuffer()
    best_eval_reward = float('-inf')
    
    logger.info(f"Starting training for {total_episodes} episodes")
    
    for episode in range(1, total_episodes + 1):
        # Collect episode experience
        episode_stats = train_episode(env, agent, buffer)
        episode_rewards.append(episode_stats['reward'])
        episode_lengths.append(episode_stats['length'])
        
        # Update agent
        if len(buffer.states) >= config.PPO_CONFIG['buffer_size']:
            data = buffer.get_arrays()
            
            # Compute advantages and returns
            advantages, returns = agent.compute_gae(
                data['rewards'].tolist(),
                data['values'].tolist(),
                data['dones'].tolist(),
                data['values'][-1]
            )
            
            # Update networks
            update_stats = agent.update(
                data['states'],
                data['actions'],
                data['log_probs'],
                advantages,
                returns
            )
            
            training_stats.append(update_stats)
            buffer.clear()
        
        # Logging
        if episode % config.TRAINING_CONFIG['log_frequency'] == 0:
            avg_reward = np.mean(episode_rewards[-config.TRAINING_CONFIG['log_frequency']:])
            logger.info(
                f"Episode {episode}/{total_episodes} | "
                f"Reward: {episode_stats['reward']:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Length: {episode_stats['length']} | "
                f"Final Power: {episode_stats['final_power']:.2e} W"
            )
        
        # Evaluation
        if episode % eval_frequency == 0:
            eval_reward = evaluate(env, agent, eval_episodes, deterministic=True)
            eval_rewards.append(eval_reward)
            logger.info(f"Evaluation at episode {episode}: Avg Reward = {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                agent.save(best_path)
                logger.info(f"New best model saved (eval reward: {eval_reward:.2f})")
        
        # Checkpointing
        if episode % save_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pt")
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    agent.save(final_path)
    logger.info(f"Final model saved: {final_path}")
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'training_stats': training_stats,
    }
    history_path = os.path.join(log_dir, "training_history.json")
    utils.save_training_history(history, history_path)
    
    # Plot training curves
    if config.VISUALIZATION_CONFIG['plot_during_training']:
        plot_path = os.path.join(log_dir, "training_curves.png") if config.VISUALIZATION_CONFIG['save_plots'] else None
        utils.plot_training_curves(
            episode_rewards,
            episode_lengths,
            eval_rewards if eval_rewards else None,
            save_path=plot_path,
            show=not config.VISUALIZATION_CONFIG['save_plots']
        )
    
    return history


def evaluate(env: WindFarmEnv, agent: PPOAgent, n_episodes: int = 10, deterministic: bool = True) -> float:
    """
    Evaluate agent
    
    Args:
        env: Environment
        agent: Agent to evaluate
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic policy
        
    Returns:
        Average reward
    """
    episode_rewards = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards)


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for wind farm control")
    parser.add_argument("--episodes", type=int, default=config.TRAINING_CONFIG['total_episodes'],
                        help="Number of training episodes")
    parser.add_argument("--wfsim-path", type=str, default=None,
                        help="Path to WFSim directory")
    parser.add_argument("--layout", type=str, default=config.LAYOUT_NAME,
                        help="Wind farm layout name")
    parser.add_argument("--checkpoint-dir", type=str, default=config.TRAINING_CONFIG['checkpoint_dir'],
                        help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default=config.TRAINING_CONFIG['log_dir'],
                        help="Directory for logs")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    utils.setup_logging(args.log_dir)
    utils.create_directories()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    logger.info("Creating environment...")
    env = WindFarmEnv(
        wfsim_path=args.wfsim_path,
        layout_name=args.layout
    )
    
    # Create agent
    logger.info("Creating PPO agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim)
    
    # Train
    try:
        history = train(
            env,
            agent,
            total_episodes=args.episodes,
            eval_frequency=config.TRAINING_CONFIG['eval_frequency'],
            eval_episodes=config.TRAINING_CONFIG['eval_episodes'],
            save_frequency=config.TRAINING_CONFIG['save_frequency'],
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
        )
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
