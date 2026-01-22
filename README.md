# PPO Wind Farm Optimization

Reinforcement learning system using Proximal Policy Optimization (PPO) to optimize wind farm power output by controlling turbine yaw angles. Integrates Python RL code with MATLAB WFSim (Wind Farm Simulator).

## Overview

This project implements a PPO-based RL agent that learns to control turbine yaw angles to maximize total wind farm power output through wake steering. The agent interacts with WFSim (MATLAB) through a Python interface, using a Gymnasium-compatible environment.

## Quick Start

**New to this project?** Check out [`QUICKSTART.md`](QUICKSTART.md) for a step-by-step guide to get started in minutes!

**Want to see it in action?** Run the example script:
```bash
python example.py
```

## Features

- **PPO Algorithm**: PyTorch implementation of Proximal Policy Optimization
- **WFSim Integration**: Seamless interface with MATLAB WFSim simulator
- **Gymnasium Environment**: Standard RL environment interface
- **Configurable Layouts**: Support for different wind farm configurations
- **Training & Evaluation**: Complete training pipeline with evaluation scripts
- **Visualization**: Plotting utilities for training curves and episode trajectories

## Requirements

### Python Dependencies

Install Python packages:
```bash
pip install -r requirements.txt
```

### MATLAB Requirements

1. **MATLAB**: MATLAB R2018b or later must be installed
2. **MATLAB Engine API for Python**: Install via:
   ```bash
   cd "matlabroot/extern/engines/python"
   python setup.py install
   ```
   Or use the `matlabengine` package (requires MATLAB to be on PATH).

3. **WFSim**: Download WFSim from [TUDelft-DataDrivenControl/WFSim](https://github.com/TUDelft-DataDrivenControl/WFSim)

## Setup

1. **Clone or download this repository**

2. **Set WFSim path**:
   - Option 1: Set environment variable:
     ```bash
     export WFSIM_PATH=/path/to/WFSim
     ```
   - Option 2: Edit `config.py` and set `WFSIM_PATH` directly

3. **Configure wind farm layout** (optional):
   - Edit `config.py` to change `LAYOUT_NAME` (default: `"sowfa_9turb_apc_alm_turbl"`)
   - Available layouts are in WFSim's `layoutDefinitions` folder

## Usage

### Training

Train a PPO agent:

```bash
python train.py --episodes 1000 --wfsim-path /path/to/WFSim --layout sowfa_9turb_apc_alm_turbl
```

Arguments:
- `--episodes`: Number of training episodes (default: 1000)
- `--wfsim-path`: Path to WFSim directory (optional if set in config/env)
- `--layout`: Wind farm layout name (default: from config)
- `--checkpoint-dir`: Directory for saving checkpoints (default: `checkpoints/`)
- `--log-dir`: Directory for logs (default: `logs/`)
- `--seed`: Random seed (default: 42)

Training outputs:
- Checkpoints saved in `checkpoints/` directory
- Training logs in `logs/` directory
- Training curves plot (if enabled in config)

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py checkpoints/best_model.pt --episodes 10 --wfsim-path /path/to/WFSim
```

Arguments:
- `model`: Path to saved model file
- `--episodes`: Number of evaluation episodes (default: 10)
- `--wfsim-path`: Path to WFSim directory
- `--layout`: Wind farm layout name
- `--output-dir`: Output directory for results (default: `evaluation_results/`)
- `--no-baseline`: Skip baseline comparison
- `--no-plots`: Don't save plots
- `--stochastic`: Use stochastic policy instead of deterministic

Evaluation outputs:
- Statistics and comparison with baseline
- Episode trajectory plots
- Power comparison plots
- Results saved as JSON

## Configuration

Edit `config.py` to customize:

- **Wind Farm**: Layout name, WFSim path
- **Environment**: Episode length, reward scaling, state/action spaces
- **PPO**: Learning rates, hyperparameters, network architecture
- **Training**: Episodes, evaluation frequency, checkpointing
- **Visualization**: Plot settings

## Project Structure

```
RL-for-Wind-Farm/
├── config.py              # Configuration file
├── matlab_interface.py    # MATLAB/WFSim interface
├── wind_farm_env.py       # Gymnasium environment
├── ppo_agent.py          # PPO agent implementation
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── utils.py              # Utility functions
├── example.py            # Simple usage example
├── test_setup.py         # Setup verification script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── QUICKSTART.md        # Quick start guide
├── TESTING.md           # Detailed testing guide
├── .gitignore           # Git ignore patterns
├── checkpoints/         # Saved models (created during training)
├── logs/                # Training logs (created during training)
└── evaluation_results/  # Evaluation outputs (created during evaluation)
```

## Key Components

### MATLAB Interface (`matlab_interface.py`)
- Manages MATLAB engine lifecycle
- Handles WFSim initialization and stepping
- Extracts state information from WFSim outputs

### Wind Farm Environment (`wind_farm_env.py`)
- Gymnasium-compatible environment
- Converts WFSim states to RL observations
- Computes rewards based on power output
- Manages episode lifecycle

### PPO Agent (`ppo_agent.py`)
- Actor-Critic networks for policy and value estimation
- PPO update with clipping
- GAE (Generalized Advantage Estimation) for advantage computation

## State and Action Spaces

**State Space** (configurable):
- Flow velocities (u, v) at turbine locations
- Individual turbine power outputs
- Current yaw angles
- Optional: Flow field regions around turbines
- Optional: Thrust coefficients

**Action Space**:
- Yaw angles for each turbine: `[-30°, +30°]` (degrees)

**Reward**:
- Total power output (normalized by baseline, if enabled)

## Troubleshooting

### MATLAB Engine Issues

- **"MATLAB engine not found"**: Ensure MATLAB is installed and on PATH
- **"Failed to start MATLAB engine"**: Check MATLAB installation and try restarting
- **Import errors**: Install MATLAB Engine API for Python from MATLAB installation directory

### WFSim Issues

- **"WFSim path not set"**: Set `WFSIM_PATH` in config.py or as environment variable
- **"Layout not found"**: Check layout name matches WFSim layout definitions
- **Simulation errors**: Verify WFSim is properly installed and MATLAB paths are correct

### Training Issues

- **Slow training**: WFSim simulation can be slow; consider reducing episode length or using fewer turbines
- **Memory issues**: Reduce buffer size or batch size in config
- **NaN values**: Check reward scaling and state normalization

## Citation

If you use this code, please cite:
- WFSim: Boersma, S., & Doekemeijer, B. (2018). WFSim: Wind Farm Simulator. Delft University of Technology. https://github.com/TUDelft-DataDrivenControl/WFSim

## License

[Add your license here]

## Contact

[Add your contact information here]
