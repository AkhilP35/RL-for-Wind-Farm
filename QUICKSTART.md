# Quick Start Guide

This guide will help you get started with the PPO Wind Farm Optimization project in 5 minutes.

## Prerequisites

1. **MATLAB** (R2018b or later) installed and on your PATH
2. **WFSim** downloaded from [TUDelft-DataDrivenControl/WFSim](https://github.com/TUDelft-DataDrivenControl/WFSim)
3. **Python 3.8+** with pip

## Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install MATLAB Engine API for Python

```bash
# Navigate to your MATLAB installation directory
cd "$(python -c 'import matlab.engine; print(matlab.engine.__file__.rsplit("/", 3)[0])')" 2>/dev/null || \
cd /Applications/MATLAB_R20XX.app/extern/engines/python

# Install the engine
python setup.py install
```

### Step 3: Configure WFSim Path

Edit `config.py` and set your WFSim path:

```python
WFSIM_PATH = "/path/to/WFSim-master"  # Update this line
```

Or set it as an environment variable:

```bash
export WFSIM_PATH=/path/to/WFSim-master
```

## Quick Test

Verify your setup is working:

```bash
python test_setup.py
```

This will test:
- ✅ Python package imports
- ✅ MATLAB engine connection
- ✅ WFSim integration
- ✅ Environment and agent creation

## Training

### Basic Training

Train a PPO agent with default settings:

```bash
python train.py --episodes 100
```

### With Custom Settings

```bash
python train.py \
  --episodes 1000 \
  --wfsim-path /path/to/WFSim \
  --layout sowfa_9turb_apc_alm_turbl \
  --checkpoint-dir my_checkpoints \
  --seed 42
```

### Monitor Training

Training progress is logged to:
- Console output (real-time)
- `logs/training.log` (detailed logs)
- `logs/training_curves.png` (visualization)
- `checkpoints/` (saved models)

## Evaluation

### Evaluate Best Model

```bash
python evaluate.py checkpoints/best_model.pt --episodes 10
```

This will:
- Run 10 evaluation episodes
- Compare with baseline (zero yaw angles)
- Save results to `evaluation_results/`
- Generate plots showing power improvement

### Evaluation Output

Results include:
- `evaluation_results.json` - Statistics and metrics
- `best_episode_trajectory.png` - Power and yaw angles over time
- `power_comparison.png` - Comparison across episodes

## Configuration

Customize behavior by editing `config.py`:

### Wind Farm Settings
```python
WFSIM_PATH = "/path/to/WFSim"
LAYOUT_NAME = "sowfa_9turb_apc_alm_turbl"
```

### Training Settings
```python
EPISODE_LENGTH = 200  # Timesteps per episode
YAW_MIN = -30.0       # Minimum yaw angle (degrees)
YAW_MAX = 30.0        # Maximum yaw angle (degrees)
```

### PPO Hyperparameters
```python
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "clip_epsilon": 0.2,
    # ... more options in config.py
}
```

## Troubleshooting

### "MATLAB engine not found"
Install MATLAB Engine API (see Step 2 above)

### "WFSim path not set"
Set `WFSIM_PATH` in `config.py` or as environment variable

### "Layout not found"
Check available layouts in `WFSim/layoutDefinitions/` and update `LAYOUT_NAME`

### Slow Training
- Reduce `EPISODE_LENGTH` in `config.py` (default: 200)
- Use a smaller layout (fewer turbines)
- WFSim simulation is computationally intensive

## Next Steps

1. **Experiment with hyperparameters** in `config.py`
2. **Try different layouts** from WFSim's layoutDefinitions
3. **Analyze results** in evaluation_results/
4. **Monitor training** with tensorboard (optional, add integration)

## Project Structure

```
RL-for-Wind-Farm/
├── config.py              # Configuration (edit this!)
├── matlab_interface.py    # MATLAB/WFSim interface
├── wind_farm_env.py       # RL environment
├── ppo_agent.py          # PPO algorithm
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── test_setup.py         # Setup verification
├── utils.py              # Utilities
└── requirements.txt      # Dependencies
```

## Help & Documentation

- **Full documentation**: See [README.md](README.md)
- **Testing guide**: See [TESTING.md](TESTING.md)
- **WFSim documentation**: [GitHub repo](https://github.com/TUDelft-DataDrivenControl/WFSim)

## Example Workflow

```bash
# 1. Verify setup
python test_setup.py

# 2. Quick training run (5 episodes)
python train.py --episodes 5

# 3. Full training (1000 episodes)
python train.py --episodes 1000

# 4. Evaluate trained model
python evaluate.py checkpoints/best_model.pt

# 5. Check results
cat evaluation_results/evaluation_results.json
```

## Key Metrics

When evaluating, look for:
- **Improvement %**: Power improvement over baseline
- **Mean Power**: Average power output during episodes
- **Yaw Angles**: Check if agent learns meaningful wake steering

A successful agent should show:
- 5-15% power improvement over baseline (typical for wake steering)
- Coordinated yaw angles (not random)
- Stable behavior during evaluation

Good luck with your wind farm optimization! 🌬️💨
