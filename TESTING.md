# Testing Guide

This guide will help you test your WFSim RL setup step by step.

## Quick Test

Run the comprehensive test script:

```bash
python test_setup.py
```

This will test:
1. ✅ Python package imports
2. ✅ MATLAB engine connection
3. ✅ WFSim path configuration
4. ✅ MATLAB interface with WFSim
5. ✅ Gymnasium environment
6. ✅ PPO agent creation

## Step-by-Step Testing

### Step 1: Test Python Packages

First, verify all required packages are installed:

```bash
pip install -r requirements.txt
```

Then test imports:
```python
python -c "import numpy, torch, gymnasium, matplotlib, matlab.engine; print('All packages OK')"
```

### Step 2: Test MATLAB Engine

Test MATLAB connection:
```python
python -c "import matlab.engine; eng = matlab.engine.start_matlab(); print('MATLAB OK:', eng.sqrt(16.0)); eng.quit()"
```

If this fails:
- Make sure MATLAB is installed
- Install MATLAB Engine API: `cd matlabroot/extern/engines/python && python setup.py install`
- Ensure MATLAB is on your PATH

### Step 3: Test WFSim Path

Verify your WFSim path in `config.py`:
```python
import config
print(config.get_wfsim_path())
```

The path should point to the WFSim directory (not a file), e.g.:
- ✅ `/Users/akhilpatel/Desktop/Dissertation/WFSim-master`
- ❌ `/Users/akhilpatel/Desktop/Dissertation/WFSim-master/WFSim_demo.m`

### Step 4: Test MATLAB Interface

Test the MATLAB interface directly:
```python
from matlab_interface import WFSimInterface
import config

wfsim = WFSimInterface(
    str(config.get_wfsim_path()),
    config.LAYOUT_NAME
)

# Test reset
state = wfsim.reset()
print(f"Reset OK: {wfsim.get_n_turbines()} turbines")

# Test step
import numpy as np
zero_yaw = np.zeros(wfsim.get_n_turbines())
state, power, done, info = wfsim.step(zero_yaw)
print(f"Step OK: Power = {power:.2e} W")

wfsim.close()
```

### Step 5: Test Environment

Test the Gymnasium environment:
```python
from wind_farm_env import WindFarmEnv
import config

env = WindFarmEnv(
    wfsim_path=str(config.get_wfsim_path()),
    layout_name=config.LAYOUT_NAME
)

# Test reset
obs, info = env.reset()
print(f"Reset OK: obs shape = {obs.shape}")

# Test step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Step OK: reward = {reward:.2f}, power = {info['power']:.2e} W")

env.close()
```

### Step 6: Test Short Training Run

Run a very short training to verify everything works:
```bash
python train.py --episodes 5 --wfsim-path /Users/akhilpatel/Desktop/Dissertation/WFSim-master
```

This will:
- Create the environment
- Initialize the agent
- Run 5 episodes
- Save checkpoints

## Common Issues

### Issue: "MATLAB engine not found"
**Solution**: Install MATLAB Engine API for Python
```bash
cd /Applications/MATLAB_R2023b.app/extern/engines/python
python setup.py install
```
(Adjust path to your MATLAB installation)

### Issue: "WFSim path not set"
**Solution**: Edit `config.py` and set `WFSIM_PATH` to the WFSim directory (not a file)

### Issue: "Layout not found"
**Solution**: Check available layouts in `WFSim-master/layoutDefinitions/` and update `LAYOUT_NAME` in `config.py`

### Issue: "InitWFSim failed"
**Solution**: 
- Verify WFSim is properly installed
- Check MATLAB paths are correct
- Try running WFSim_demo.m in MATLAB first to verify it works

### Issue: Slow performance
**Solution**: 
- This is normal - WFSim simulation is computationally intensive
- For testing, reduce `EPISODE_LENGTH` in `config.py` to 50-100 steps
- Consider using a smaller layout (e.g., 2-turbine layout) for initial testing

## Next Steps

Once all tests pass:

1. **Start training**: `python train.py --episodes 1000`
2. **Monitor progress**: Check `logs/training.log` and `logs/training_curves.png`
3. **Evaluate model**: `python evaluate.py checkpoints/best_model.pt`

## Getting Help

If tests fail:
1. Check the error messages carefully
2. Verify MATLAB and WFSim are working independently
3. Check that all paths are correct
4. Review the logs in `logs/` directory
