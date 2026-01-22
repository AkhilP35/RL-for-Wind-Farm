#!/usr/bin/env python3
"""
Simple example demonstrating the PPO Wind Farm Optimization system

This script shows how to:
1. Create a wind farm environment
2. Initialize a PPO agent
3. Run a few episodes with random actions
4. Display results

This is useful for testing your setup without running full training.
"""

import numpy as np
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("PPO Wind Farm Optimization - Simple Example")
    logger.info("=" * 60)
    
    # Import components
    try:
        from wind_farm_env import WindFarmEnv
        from ppo_agent import PPOAgent
        import config
    except ImportError as e:
        logger.error(f"Failed to import components: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    
    # Check WFSim path
    try:
        wfsim_path = config.get_wfsim_path()
        logger.info(f"WFSim path: {wfsim_path}")
    except ValueError as e:
        logger.error(str(e))
        logger.error("Please set WFSIM_PATH environment variable or update config.py")
        return 1
    
    # Create environment
    logger.info("\nCreating wind farm environment...")
    try:
        env = WindFarmEnv()
        logger.info(f"✓ Environment created successfully")
        logger.info(f"  - Number of turbines: {env.n_turbines}")
        logger.info(f"  - Observation space: {env.observation_space.shape}")
        logger.info(f"  - Action space: {env.action_space.shape}")
        logger.info(f"  - Episode length: {env.episode_length}")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        logger.error("Make sure MATLAB and WFSim are properly configured")
        return 1
    
    # Create agent
    logger.info("\nCreating PPO agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim)
    logger.info(f"✓ Agent created successfully")
    
    # Run a few episodes with random actions
    n_episodes = 3
    logger.info(f"\nRunning {n_episodes} episodes with random actions...")
    logger.info("(This is just to test the system, not actual training)")
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        logger.info(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        # Reset environment
        obs, info = env.reset()
        logger.info(f"Environment reset. Baseline power: {info.get('baseline_power', 0):.2e} W")
        
        episode_reward = 0
        step_count = 0
        powers = []
        
        # Run episode
        done = False
        while not done and step_count < 10:  # Limit to 10 steps for demo
            # Get action from agent (random at first since not trained)
            action, _, _ = agent.select_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            powers.append(step_info['power'])
            step_count += 1
            
            # Log every step for first episode
            if episode == 0:
                logger.info(
                    f"  Step {step_count}: "
                    f"Power={step_info['power']:.2e} W, "
                    f"Reward={reward:.4f}, "
                    f"Yaw angles={step_info['yaw_angles']}"
                )
        
        episode_rewards.append(episode_reward)
        avg_power = np.mean(powers)
        
        logger.info(
            f"Episode {episode + 1} complete: "
            f"Total reward={episode_reward:.2f}, "
            f"Steps={step_count}, "
            f"Avg power={avg_power:.2e} W"
        )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Episodes run: {n_episodes}")
    logger.info(f"Average reward: {np.mean(episode_rewards):.2f}")
    logger.info(f"Reward range: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
    
    # Cleanup
    env.close()
    logger.info("\n✓ Example completed successfully!")
    logger.info("\nNext steps:")
    logger.info("  1. Run full training: python train.py --episodes 100")
    logger.info("  2. Evaluate trained model: python evaluate.py checkpoints/best_model.pt")
    logger.info("  3. See QUICKSTART.md for more details")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)
