"""
Configuration file for PPO Wind Farm Optimization
Centralized hyperparameters and settings
"""

import os
from pathlib import Path

# Wind Farm Configuration
WFSIM_PATH = "/Users/akhilpatel/Desktop/Dissertation/WFSim-master"  # Path to WFSim directory (user must set this)
LAYOUT_NAME = "sowfa_9turb_apc_alm_turbl"  # Default layout, can be changed
MODEL_OPTIONS = "solverSet_default"  # Solver options function name

# Environment Parameters
EPISODE_LENGTH = 200  # Number of timesteps per episode
MAX_EPISODE_LENGTH = 500  # Maximum episode length
REWARD_SCALE = 1.0  # Scaling factor for rewards
USE_NORMALIZED_REWARDS = True  # Normalize rewards by baseline power

# Action Space (Yaw Angles)
YAW_MIN = -30.0  # Minimum yaw angle in degrees
YAW_MAX = 30.0   # Maximum yaw angle in degrees
YAW_RATE_LIMIT = None  # Maximum yaw rate change per step (None = no limit)

# State Space Configuration
INCLUDE_FLOW_FIELD = True  # Include full flow field data in state
FLOW_FIELD_REGION_SIZE = 5  # Size of region around each turbine to extract (grid points)
INCLUDE_TURBINE_POWER = True  # Include individual turbine power outputs
INCLUDE_CURRENT_YAW = True  # Include current yaw angles in state
INCLUDE_THRUST_COEFF = False  # Include thrust coefficients in state

# PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,  # Discount factor
    "lambda": 0.95,  # GAE lambda
    "clip_epsilon": 0.2,  # PPO clipping parameter
    "entropy_coef": 0.01,  # Entropy bonus coefficient
    "value_coef": 0.5,  # Value loss coefficient
    "max_grad_norm": 0.5,  # Gradient clipping
    "update_epochs": 10,  # Number of update epochs per batch
    "batch_size": 64,  # Batch size for updates
    "buffer_size": 2048,  # Size of experience buffer
}

# Neural Network Architecture
NETWORK_CONFIG = {
    "actor_hidden_sizes": [256, 256],  # Actor network hidden layers
    "critic_hidden_sizes": [256, 256],  # Critic network hidden layers
    "activation": "relu",  # Activation function
}

# Training Parameters
TRAINING_CONFIG = {
    "total_episodes": 1000,  # Total number of training episodes
    "eval_frequency": 50,  # Evaluate every N episodes
    "eval_episodes": 10,  # Number of episodes for evaluation
    "save_frequency": 100,  # Save checkpoint every N episodes
    "log_frequency": 10,  # Log metrics every N episodes
    "checkpoint_dir": "checkpoints",  # Directory for saving models
    "log_dir": "logs",  # Directory for logs
}

# MATLAB Engine Settings
MATLAB_CONFIG = {
    "startup_timeout": 30,  # Timeout for MATLAB engine startup (seconds)
    "background": False,  # Run MATLAB in background
}

# Visualization
VISUALIZATION_CONFIG = {
    "plot_during_training": True,  # Show plots during training
    "save_plots": True,  # Save plots to files
    "plot_frequency": 50,  # Plot every N episodes
}

# Random Seed
RANDOM_SEED = 42

def get_wfsim_path():
    """Get WFSim path, checking environment variable if not set in config"""
    if WFSIM_PATH:
        return Path(WFSIM_PATH)
    wfsim_env = os.getenv("WFSIM_PATH")
    if wfsim_env:
        return Path(wfsim_env)
    raise ValueError(
        "WFSim path not set! Please set WFSIM_PATH in config.py or as environment variable"
    )
