# Quickstart Guide

This guide will help you get started with the PPO Wind Farm Optimization system in just a few minutes.

## Prerequisites

1. **MATLAB** (R2018b or later)
2. **WFSim** - Download from [TUDelft-DataDrivenControl/WFSim](https://github.com/TUDelft-DataDrivenControl/WFSim)
3. **Python** 3.8 or later

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install MATLAB Engine API for Python

Find your MATLAB installation directory and run:

```bash
# macOS/Linux
cd /Applications/MATLAB_R2023b.app/extern/engines/python  # Adjust to your MATLAB version
python setup.py install

# Windows
cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"  # Adjust to your MATLAB version
python setup.py install
```

### 3. Set WFSim Path

Set the environment variable to point to your WFSim directory:

```bash
# macOS/Linux
export WFSIM_PATH=/path/to/WFSim

# Windows (PowerShell)
$env:WFSIM_PATH = "C:\path\to\WFSim"

# Windows (Command Prompt)
set WFSIM_PATH=C:\path\to\WFSim
```

Or edit `config.py` and set `WFSIM_PATH` directly:
```python
WFSIM_PATH = "/path/to/WFSim"
```

## Quick Test

Verify your setup:

```bash
python test_setup.py
```

This will test all components. All tests should pass before training.

## Training

### Simple Training (Default Settings)

```bash
python train.py --episodes 100
```

### Training with Custom Settings

```bash
python train.py \
  --episodes 1000 \
  --wfsim-path /path/to/WFSim \
  --layout sowfa_9turb_apc_alm_turbl \
  --checkpoint-dir my_checkpoints \
  --log-dir my_logs
```

### Training Output

Training will create:
- `checkpoints/` - Saved models
- `logs/` - Training logs and plots
- `logs/training.log` - Detailed log file
- `logs/training_curves.png` - Training progress plots

## Evaluation

Evaluate a trained model:

```bash
python evaluate.py checkpoints/best_model.pt --episodes 10
```

This will:
- Run 10 evaluation episodes
- Compare with baseline (zero yaw angles)
- Generate plots showing:
  - Power output over time
  - Yaw angle trajectories
  - Performance comparison
- Save results to `evaluation_results/`

## Configuration

Customize settings in `config.py`:

### Environment Settings
```python
EPISODE_LENGTH = 200          # Steps per episode
YAW_MIN = -30.0              # Minimum yaw angle (degrees)
YAW_MAX = 30.0               # Maximum yaw angle (degrees)
```

### PPO Hyperparameters
```python
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "clip_epsilon": 0.2,
    # ... more settings
}
```

### Network Architecture
```python
NETWORK_CONFIG = {
    "actor_hidden_sizes": [256, 256],
    "critic_hidden_sizes": [256, 256],
}
```

## Example: Quick Training Run

Here's a complete example for a quick test:

```bash
# 1. Verify setup
python test_setup.py

# 2. Train for a few episodes (quick test)
python train.py --episodes 5

# 3. Check the logs
cat logs/training.log

# 4. Evaluate the model (if training completed)
python evaluate.py checkpoints/final_model.pt --episodes 3
```

## Monitoring Training

### Watch Training Progress
```bash
# View live training log
tail -f logs/training.log
```

### View Training Curves
Open `logs/training_curves.png` to see:
- Episode rewards over time
- Episode lengths
- Moving average

## Tips

1. **Start Small**: Begin with fewer episodes (e.g., 100) to verify everything works
2. **Episode Length**: Reduce `EPISODE_LENGTH` in `config.py` for faster testing
3. **Layout**: Use a smaller layout (fewer turbines) for faster training
4. **GPU**: The code automatically uses GPU if available via PyTorch
5. **Checkpoints**: Best model is saved automatically based on evaluation performance

## Common Issues

### "MATLAB engine not found"
- Install MATLAB Engine API (see Installation step 2)
- Ensure MATLAB is on your PATH

### "WFSim path not set"
- Set `WFSIM_PATH` environment variable
- Or edit `config.py` directly

### Slow Training
- This is normal - WFSim simulation is computationally intensive
- Reduce `EPISODE_LENGTH` for faster iterations
- Use fewer turbines for initial testing

## Next Steps

Once training is complete:

1. **Visualize Results**: Check `evaluation_results/` for plots
2. **Tune Hyperparameters**: Adjust `config.py` for better performance
3. **Try Different Layouts**: Experiment with other wind farm configurations
4. **Extended Training**: Increase episodes for better convergence

## Getting Help

- Check `TESTING.md` for detailed testing instructions
- Review `README.md` for complete documentation
- Check logs in `logs/training.log` for debugging

## Project Structure

```
RL-for-Wind-Farm/
├── config.py              # Configuration
├── matlab_interface.py    # MATLAB/WFSim interface
├── wind_farm_env.py       # RL environment
├── ppo_agent.py          # PPO implementation
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── utils.py              # Utilities
├── test_setup.py         # Setup verification
├── requirements.txt      # Python dependencies
└── README.md            # Full documentation
```
