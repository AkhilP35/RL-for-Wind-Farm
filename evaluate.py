"""
Evaluation script for trained PPO agent
"""

import numpy as np
import torch
import argparse
import os
from pathlib import Path
from typing import List, Dict
import logging
import matplotlib.pyplot as plt

import config
from wind_farm_env import WindFarmEnv
from ppo_agent import PPOAgent
import utils

logger = logging.getLogger(__name__)


def evaluate_episode(
    env: WindFarmEnv,
    agent: PPOAgent,
    deterministic: bool = True,
    render: bool = False
) -> Dict:
    """
    Evaluate agent for one episode
    
    Returns:
        Dictionary with episode statistics
    """
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    
    powers = []
    yaw_angles = []
    power_per_turbine_history = []
    
    done = False
    while not done:
        # Select action
        action, _, _ = agent.select_action(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        
        # Record data
        powers.append(step_info['power'])
        yaw_angles.append(step_info['yaw_angles'].copy())
        power_per_turbine_history.append(step_info['power_per_turbine'].copy())
        
        episode_reward += reward
        episode_length += 1
    
    return {
        'reward': episode_reward,
        'length': episode_length,
        'powers': powers,
        'yaw_angles': yaw_angles,
        'power_per_turbine_history': power_per_turbine_history,
        'baseline_power': info.get('baseline_power', 0.0),
        'final_power': powers[-1] if powers else 0.0,
        'mean_power': np.mean(powers) if powers else 0.0,
        'max_power': np.max(powers) if powers else 0.0,
    }


def compare_with_baseline(env: WindFarmEnv, n_episodes: int = 10) -> Dict:
    """
    Evaluate baseline (zero yaw angles) performance
    
    Returns:
        Dictionary with baseline statistics
    """
    logger.info("Evaluating baseline (zero yaw angles)...")
    
    baseline_powers = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        baseline_power = info.get('baseline_power', 0.0)
        
        # Run episode with zero yaw angles
        done = False
        episode_powers = []
        while not done:
            zero_action = np.zeros(env.action_space.shape[0])
            obs, _, terminated, truncated, step_info = env.step(zero_action)
            done = terminated or truncated
            episode_powers.append(step_info['power'])
        
        baseline_powers.extend(episode_powers)
    
    return {
        'mean_power': np.mean(baseline_powers),
        'std_power': np.std(baseline_powers),
        'max_power': np.max(baseline_powers),
        'min_power': np.min(baseline_powers),
    }


def evaluate(
    model_path: str,
    wfsim_path: str = None,
    layout_name: str = None,
    n_episodes: int = 10,
    deterministic: bool = True,
    compare_baseline: bool = True,
    save_plots: bool = True,
    output_dir: str = "evaluation_results",
):
    """
    Evaluate trained agent
    
    Args:
        model_path: Path to saved model
        wfsim_path: Path to WFSim directory
        layout_name: Layout name
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic policy
        compare_baseline: Whether to compare with baseline
        save_plots: Whether to save plots
        output_dir: Output directory for results
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    utils.setup_logging(output_dir)
    
    # Create environment
    logger.info("Creating environment...")
    env = WindFarmEnv(
        wfsim_path=wfsim_path,
        layout_name=layout_name or config.LAYOUT_NAME
    )
    
    # Create agent and load model
    logger.info(f"Loading model from {model_path}...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim)
    agent.load(model_path)
    
    # Evaluate agent
    logger.info(f"Evaluating agent for {n_episodes} episodes...")
    episode_results = []
    
    for episode in range(n_episodes):
        logger.info(f"Episode {episode + 1}/{n_episodes}")
        result = evaluate_episode(env, agent, deterministic=deterministic)
        episode_results.append(result)
        
        logger.info(
            f"  Reward: {result['reward']:.2f} | "
            f"Length: {result['length']} | "
            f"Mean Power: {result['mean_power']:.2e} W | "
            f"Max Power: {result['max_power']:.2e} W"
        )
    
    # Compute statistics
    rewards = [r['reward'] for r in episode_results]
    mean_powers = [r['mean_power'] for r in episode_results]
    max_powers = [r['max_power'] for r in episode_results]
    
    stats = {
        'reward': utils.compute_statistics(rewards),
        'mean_power': utils.compute_statistics(mean_powers),
        'max_power': utils.compute_statistics(max_powers),
    }
    
    logger.info("\n=== Evaluation Statistics ===")
    logger.info(f"Reward - Mean: {stats['reward']['mean']:.2f}, Std: {stats['reward']['std']:.2f}")
    logger.info(f"Mean Power - Mean: {stats['mean_power']['mean']:.2e} W, Std: {stats['mean_power']['std']:.2e} W")
    logger.info(f"Max Power - Mean: {stats['max_power']['mean']:.2e} W, Std: {stats['max_power']['std']:.2e} W")
    
    # Compare with baseline
    if compare_baseline:
        baseline_stats = compare_with_baseline(env, n_episodes=n_episodes)
        logger.info("\n=== Baseline Statistics ===")
        logger.info(f"Mean Power: {baseline_stats['mean_power']:.2e} W")
        logger.info(f"Std Power: {baseline_stats['std_power']:.2e} W")
        
        improvement = ((stats['mean_power']['mean'] - baseline_stats['mean_power']) / 
                      baseline_stats['mean_power'] * 100)
        logger.info(f"\nImprovement over baseline: {improvement:.2f}%")
        
        stats['baseline'] = baseline_stats
        stats['improvement_percent'] = improvement
    
    # Plot results
    if save_plots:
        # Plot best episode trajectory
        best_episode_idx = np.argmax([r['reward'] for r in episode_results])
        best_result = episode_results[best_episode_idx]
        
        plot_path = os.path.join(output_dir, "best_episode_trajectory.png")
        utils.plot_episode_trajectory(
            best_result['powers'],
            best_result['yaw_angles'],
            baseline_power=best_result['baseline_power'],
            save_path=plot_path,
            show=False
        )
        
        # Plot power comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        episodes = range(1, n_episodes + 1)
        ax.plot(episodes, mean_powers, 'o-', label='Agent Mean Power', markersize=6)
        if compare_baseline:
            ax.axhline(y=baseline_stats['mean_power'], color='r', linestyle='--', 
                      label=f'Baseline ({baseline_stats["mean_power"]:.2e} W)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Power (W)')
        ax.set_title('Evaluation: Mean Power per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "power_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save results
    import json
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON
        results_json = {
            'stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                         for kk, vv in v.items()} 
                     for k, v in stats.items()},
            'episode_results': [
                {
                    'reward': float(r['reward']),
                    'length': r['length'],
                    'mean_power': float(r['mean_power']),
                    'max_power': float(r['max_power']),
                    'final_power': float(r['final_power']),
                }
                for r in episode_results
            ]
        }
        json.dump(results_json, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    
    env.close()
    
    return stats, episode_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("model", type=str, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--wfsim-path", type=str, default=None,
                        help="Path to WFSim directory")
    parser.add_argument("--layout", type=str, default=config.LAYOUT_NAME,
                        help="Wind farm layout name")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Don't compare with baseline")
    parser.add_argument("--no-plots", action="store_true",
                        help="Don't save plots")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        wfsim_path=args.wfsim_path,
        layout_name=args.layout,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        compare_baseline=not args.no_baseline,
        save_plots=not args.no_plots,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
