# Testing in Cursor IDE

This guide walks you through setting up and testing this wind farm RL project in Cursor IDE.

## Prerequisites

Before starting, ensure you have:
1. **MATLAB** (R2018b or later) installed
2. **WFSim** downloaded from [TUDelft-DataDrivenControl/WFSim](https://github.com/TUDelft-DataDrivenControl/WFSim)
3. **Python 3.8+** installed
4. **Cursor IDE** installed

## Step 1: Clone/Open the Repository

In Cursor IDE:
1. Open Cursor
2. Go to `File` → `Open Folder` (or `File` → `Open` on macOS)
3. Navigate to this repository folder
4. Click "Select Folder" / "Open"

## Step 2: Set Up Python Environment in Cursor

### Option A: Use Cursor's Built-in Terminal

1. Open the integrated terminal in Cursor:
   - Press `` Ctrl+` `` (Windows/Linux) or `` Cmd+` `` (macOS)
   - Or go to `View` → `Terminal`

2. Create a virtual environment (recommended):
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option B: Use Existing Python Environment

If you already have a Python environment with the required packages:

1. In Cursor, open the Command Palette:
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
   
2. Type "Python: Select Interpreter"

3. Choose your Python interpreter with the installed packages

## Step 3: Install MATLAB Engine API

The MATLAB Engine API needs to be installed in your Python environment:

1. In Cursor's terminal, navigate to MATLAB's Python engine directory:

   **Windows:**
   ```bash
   cd "C:\Program Files\MATLAB\R20XX\extern\engines\python"
   ```

   **macOS:**
   ```bash
   cd /Applications/MATLAB_R20XX.app/extern/engines/python
   ```

   **Linux:**
   ```bash
   cd /usr/local/MATLAB/R20XX/extern/engines/python
   ```

   Replace `R20XX` with your MATLAB version (e.g., `R2023b`)

2. Install the engine:
   ```bash
   python setup.py install
   ```

## Step 4: Configure WFSim Path

1. In Cursor, open `config.py`
2. Find the line with `WFSIM_PATH`
3. Update it to point to your WFSim installation:
   ```python
   WFSIM_PATH = "/path/to/your/WFSim-master"
   ```

   **Examples:**
   - Windows: `"C:/Users/YourName/Documents/WFSim-master"`
   - macOS: `"/Users/YourName/Desktop/WFSim-master"`
   - Linux: `"/home/YourName/WFSim-master"`

4. Save the file (`Ctrl+S` or `Cmd+S`)

## Step 5: Run the Setup Test

Now test your configuration in Cursor:

1. In Cursor's terminal, run:
   ```bash
   python test_setup.py
   ```

2. The test will verify:
   - ✅ Python packages (numpy, torch, gymnasium, matplotlib)
   - ✅ MATLAB engine connection
   - ✅ WFSim path and files
   - ✅ MATLAB interface with WFSim
   - ✅ Environment creation
   - ✅ PPO agent initialization

3. Expected output:
   ```
   ============================================================
   WFSim RL Setup Test
   ============================================================
   Testing Python package imports...
   ✓ numpy 1.26.0
   ✓ torch 2.6.0
   ✓ gymnasium 0.28.1
   ✓ matplotlib 3.8.0
   ✓ matlab.engine
   
   Testing MATLAB engine connection...
   ✓ MATLAB engine working (sqrt(16) = 4.0)
   
   Testing WFSim path...
   ✓ WFSim path: /path/to/WFSim-master
   ✓ WFSim directory structure looks good
   
   Testing MATLAB interface with WFSim...
   ✓ Reset successful (n_turbines: 9)
   ✓ Step successful (power: 1.23e+07 W)
   ...
   
   All tests passed! You're ready to start training.
   ```

## Step 6: Run a Quick Training Test

Test training for just a few episodes:

```bash
python train.py --episodes 5
```

This will:
- Initialize the environment and agent
- Run 5 training episodes
- Save checkpoints to `checkpoints/`
- Log progress to console and `logs/training.log`

## Debugging in Cursor

### Run Python Files with Debugging

1. Open a Python file (e.g., `train.py`)
2. Click on the left margin next to a line number to set a breakpoint
3. Press `F5` or go to `Run` → `Start Debugging`
4. Choose "Python File" when prompted

### View Variables and Stack

When debugging:
- **Variables panel**: Shows all variables in current scope
- **Call Stack**: Shows function call hierarchy
- **Debug Console**: Execute Python code in current context

### Common Issues in Cursor

#### Issue: "No module named 'matlab.engine'"
**Solution:** 
- Ensure MATLAB Engine API is installed in the active Python environment
- Verify the correct Python interpreter is selected in Cursor
- Restart Cursor after installing MATLAB Engine API

#### Issue: "MATLAB engine not found"
**Solution:**
- Verify MATLAB is installed and on your system PATH
- Try restarting your computer after MATLAB installation
- Check MATLAB license is valid

#### Issue: "WFSim path not set"
**Solution:**
- Update `WFSIM_PATH` in `config.py`
- Use absolute paths (not relative)
- Ensure the path exists and contains WFSim files

## Using Cursor's AI Features

Cursor has built-in AI that can help:

1. **Ctrl+K** (Cmd+K on macOS): Edit code with AI
   - Select code and press Ctrl+K
   - Ask AI to explain, modify, or fix code

2. **Ctrl+L** (Cmd+L on macOS): Chat with AI about the code
   - Ask questions about the implementation
   - Get explanations of how components work

3. **Tab**: Use AI autocomplete
   - Start typing and press Tab to accept AI suggestions

## Running Specific Tests

### Test Only Imports
```python
python -c "import config, matlab_interface, wind_farm_env, ppo_agent; print('✓ All imports OK')"
```

### Test MATLAB Connection Only
```python
python -c "import matlab.engine; eng = matlab.engine.start_matlab(); print('MATLAB OK'); eng.quit()"
```

### Test Environment Only
```python
from wind_farm_env import WindFarmEnv
import config

env = WindFarmEnv(
    wfsim_path=str(config.get_wfsim_path()),
    layout_name=config.LAYOUT_NAME
)
obs, info = env.reset()
print(f"✓ Environment working, obs shape: {obs.shape}")
env.close()
```

## Cursor Terminal Tips

### Split Terminal
- Right-click in terminal area → "Split Terminal"
- Run tests in one terminal while monitoring logs in another

### Terminal Profiles
- Cursor remembers your terminal sessions
- Create different profiles for training, testing, evaluation

### Clear Terminal
- Type `clear` (macOS/Linux) or `cls` (Windows)
- Or press `Ctrl+L` in terminal

## Next Steps

After setup works:

1. **Experiment with small changes:**
   - Modify hyperparameters in `config.py`
   - Run short training sessions (5-10 episodes)
   - Check results in `logs/` and `checkpoints/`

2. **Use Cursor's file navigation:**
   - `Ctrl+P` (Cmd+P): Quick file open
   - `Ctrl+Shift+F` (Cmd+Shift+F): Search across files
   - `F12`: Go to definition
   - `Alt+←/→`: Navigate backward/forward

3. **Monitor training:**
   - Keep `logs/training.log` open in Cursor
   - File auto-updates as training progresses
   - Use `tail -f logs/training.log` in terminal for live updates

4. **Visualize results:**
   - Cursor can preview images (`.png` files)
   - Click on generated plots to view them
   - Training curves appear in `logs/training_curves.png`

## Useful Cursor Shortcuts

| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Command Palette | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| Quick Open File | `Ctrl+P` | `Cmd+P` |
| Toggle Terminal | ``Ctrl+` `` | ``Cmd+` `` |
| Run/Debug | `F5` | `F5` |
| Toggle Sidebar | `Ctrl+B` | `Cmd+B` |
| Find in Files | `Ctrl+Shift+F` | `Cmd+Shift+F` |
| Go to Definition | `F12` | `F12` |
| AI Edit | `Ctrl+K` | `Cmd+K` |
| AI Chat | `Ctrl+L` | `Cmd+L` |

## Getting Help

If you encounter issues:

1. Check `TESTING.md` for detailed troubleshooting
2. Review `QUICKSTART.md` for setup instructions
3. Read `IMPLEMENTATION_SUMMARY.md` for technical details
4. Use Cursor's AI (Ctrl+L) to ask about specific errors

## Example: Complete First Run in Cursor

Here's a complete workflow from opening Cursor to running your first training:

```bash
# 1. Open terminal in Cursor (Ctrl+`)

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install MATLAB Engine (navigate to MATLAB directory first)
cd /Applications/MATLAB_R2023b.app/extern/engines/python
python setup.py install
cd -  # Return to project directory

# 5. Update config.py (edit WFSIM_PATH)

# 6. Test setup
python test_setup.py

# 7. Run quick test training
python train.py --episodes 5

# 8. View results
ls -la checkpoints/
cat logs/training.log
```

That's it! You're now set up to develop and test the wind farm RL agent in Cursor IDE. Happy coding! 🚀
