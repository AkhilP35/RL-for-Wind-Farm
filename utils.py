"""
Utility functions for wind farm RL
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "training.log")
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_training_history(history: Dict, filepath: str):
    """Save training history to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    history_json = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            history_json[key] = value.tolist()
        elif isinstance(value, list):
            history_json[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
        else:
            history_json[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(history_json, f, indent=2)
    
    logger.info(f"Training history saved to {filepath}")


def load_training_history(filepath: str) -> Dict:
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(
    episode_rewards: List[float],
    episode_lengths: List[int],
    eval_rewards: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        eval_rewards: Optional list of evaluation rewards
        save_path: Path to save plot
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    axes[0].plot(episode_rewards, alpha=0.6, label='Training')
    if eval_rewards:
        eval_episodes = np.linspace(0, len(episode_rewards) - 1, len(eval_rewards))
        axes[0].plot(eval_episodes, eval_rewards, 'o-', label='Evaluation', markersize=4)
    
    # Moving average
    if len(episode_rewards) > 10:
        window = min(50, len(episode_rewards) // 10)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[1].plot(episode_lengths, alpha=0.6)
    if len(episode_lengths) > 10:
        window = min(50, len(episode_lengths) // 10)
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(episode_lengths)), moving_avg, 'r-', linewidth=2)
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('Episode Lengths')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_episode_trajectory(
    powers: List[float],
    yaw_angles: List[np.ndarray],
    baseline_power: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot episode trajectory showing power and yaw angles over time
    
    Args:
        powers: List of power outputs at each timestep
        yaw_angles: List of yaw angle arrays at each timestep
        baseline_power: Baseline power for comparison
        save_path: Path to save plot
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    timesteps = range(len(powers))
    
    # Plot power
    axes[0].plot(timesteps, powers, 'b-', linewidth=2, label='Total Power')
    if baseline_power is not None:
        axes[0].axhline(y=baseline_power, color='r', linestyle='--', label='Baseline Power')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Power (W)')
    axes[0].set_title('Power Output Over Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot yaw angles
    yaw_array = np.array(yaw_angles)
    n_turbines = yaw_array.shape[1]
    for i in range(n_turbines):
        axes[1].plot(timesteps, yaw_array[:, i], label=f'Turbine {i+1}', alpha=0.7)
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Yaw Angle (degrees)')
    axes[1].set_title('Yaw Angles Over Episode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values"""
    values_array = np.array(values)
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array)),
    }


def create_directories(base_dir: str = "."):
    """Create necessary directories for training"""
    dirs = [
        os.path.join(base_dir, "checkpoints"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "plots"),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs
