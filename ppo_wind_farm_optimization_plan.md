---
name: PPO Wind Farm Optimization
overview: Build a complete PPO-based RL system in Python to optimize wind farm power output by controlling turbine yaw angles, integrating with MATLAB WFSim through a Gymnasium-compatible environment.
todos: []
---

# PPO Wind Farm Optimization System

## Architecture Overview

The system will consist of three main components:

1. **MATLAB Interface Layer** (`matlab_interface.py`): Wraps WFSim MATLAB calls
2. **Gymnasium Environment** (`wind_farm_env.py`): Implements OpenAI Gym interface for RL training
3. **PPO Agent** (`ppo_agent.py`): PyTorch-based PPO implementation with training loop
```
Python RL Code → Gymnasium Environment → MATLAB Interface → WFSim (MATLAB)
     ↑                                                              ↓
     └─────────────────── State/Observation ←──────────────────────┘
```


## Data Flow and Workflow

### Initialization:

1. Start MATLAB engine
2. Add WFSim paths (`layoutDefinitions`, `controlDefinitions`, `solverDefinitions`)
3. Call `layoutSet_*()` to create `Wp` struct for chosen layout
4. Call `InitWFSim(Wp, modelOptions)` to initialize simulation → returns `sol`, `sys`
5. Extract initial state from `sol` struct

### Episode Loop:

1. **Agent selects action**: Yaw angles `phi` for all turbines (numpy array, shape `[n_turbines]`)
2. **Environment receives action**: Convert to MATLAB format, create `turbInput` struct:
   ```python
   turbInput.phi = matlab.double(yaw_angles.tolist())  # Convert to MATLAB array
   turbInput.CT_prime = baseline_CT_prime  # Keep at default values
   ```

3. **MATLAB Interface calls WFSim**: `WFSim_timestepping(sol, sys, Wp, turbInput, modelOptions)`
4. **WFSim returns updated state**: New `sol` struct with updated flow fields and turbine outputs
5. **Extract observation**: From `sol.u`, `sol.v`, `sol.turbine.power`, `sol.turbine.phi`
6. **Compute reward**: Based on total power output (`sum(sol.turbine.power)`)
7. **Return to agent**: `(observation, reward, done, info)`

### State Extraction Example:

```python
# Extract flow velocities at turbine locations
u_at_turbines = [sol.u[turbine_i, turbine_j] for each turbine]
v_at_turbines = [sol.v[turbine_i, turbine_j] for each turbine]

# Extract power outputs
power_outputs = sol.turbine.power  # Already available per turbine

# Extract current yaw angles
current_yaw = sol.turbine.phi

# Combine into observation vector
observation = [u1, v1, power1, yaw1, u2, v2, power2, yaw2, ...]
```

## Component Details

### 1. MATLAB Interface (`matlab_interface.py`)

**Purpose**: Bridge between Python and MATLAB WFSim

**Key Functions**:

- `__init__(wfsim_path, layout_name)`: Start MATLAB engine, add WFSim paths, initialize layout
- `reset()`: Call `InitWFSim()` to reset wind farm to initial state, return initial `sol` struct
- `step(yaw_angles)`: Create `turbInput` struct with phi values, call `WFSim_timestepping()`, return updated `sol`
- `get_state()`: Extract state from `sol` struct:
  - `sol.u`, `sol.v`: Flow field components (full mesh or at turbine locations)
  - `sol.turbine.power`: Individual turbine power outputs
  - `sol.turbine.phi`: Current yaw angles
  - `sol.turbine.CT`: Thrust coefficients
  - `sol.time`, `sol.k`: Current time and timestep
- `get_power_output()`: Extract total/individual power from `sol.turbine.power`
- `close()`: Clean up MATLAB engine

**WFSim Integration Details**:

- Uses actual WFSim functions: `InitWFSim()`, `WFSim_timestepping()`
- Works with WFSim structs: `Wp` (wind farm settings), `sol` (solution states), `sys` (system matrices)
- Control via `turbInput` struct:
  - `turbInput.phi`: Yaw angles (degrees) - **primary control variable**
  - `turbInput.CT_prime`: Thrust coefficient (kept at default/baseline values)
  - For baseline CT_prime: Use values from `controlSet_*()` or compute from `Wp.turbine` settings
  - CT_prime remains constant during episodes (only phi changes)
- Layout configuration: Support for different layouts via `layoutSet_*()` functions
- MATLAB paths: Add `layoutDefinitions`, `controlDefinitions`, `solverDefinitions` folders

**State Extraction**:

- Flow field: `sol.u`, `sol.v` matrices (can extract at turbine locations or full field)
- Turbine outputs: `sol.turbine.power`, `sol.turbine.CT`, `sol.turbine.phi`
- Flow field features: Extract relevant regions around turbines for wake information

**Dependencies**: `matlab.engine` (MATLAB Engine API for Python)

### 2. Wind Farm Environment (`wind_farm_env.py`)

**Purpose**: Gymnasium-compatible environment for RL training

**Class**: `WindFarmEnv(gym.Env)`

**Observation Space**:

- Type: `Box` (continuous)
- Shape: `(n_turbines * state_features,)` where state_features includes:
  - Wind speed components (u, v) at each turbine location (from `sol.u`, `sol.v`)
  - Current power output (normalized, from `sol.turbine.power`)
  - Current yaw angle (normalized, from `sol.turbine.phi`)
  - Flow field features: Extract wake information from flow field around turbines
  - Optional: Thrust coefficient (`sol.turbine.CT`) for additional context

**Action Space**:

- Type: `Box` (continuous)
- Shape: `(n_turbines,)`
- Range: `[-30°, +30°]` (yaw angles in degrees, converted to radians if needed)

**Methods**:

- `reset()`: Initialize/reset simulation, return initial observation
- `step(action)`: Apply yaw angles, advance simulation, return (obs, reward, done, info)
- `render()`: Optional visualization
- `close()`: Cleanup

**Reward Function**:

- Primary: Total power output (normalized or relative improvement)
- Optional: Penalty for large yaw angle changes (smooth control)
- Optional: Penalty for exceeding constraints

**Episode Structure**:

- Fixed episode length (e.g., 100-500 timesteps)
- Or until convergence/termination condition

### 3. PPO Agent (`ppo_agent.py`)

**Purpose**: PyTorch implementation of Proximal Policy Optimization

**Components**:

**Actor Network** (`Actor`):

- Input: State vector
- Output: Mean and std for action distribution (Gaussian)
- Architecture: Multi-layer MLP with ReLU activations
- Output layer: Tanh activation scaled to action range

**Critic Network** (`Critic`):

- Input: State vector
- Output: Value estimate (scalar)
- Architecture: Multi-layer MLP with ReLU activations

**PPO Agent Class** (`PPOAgent`):

- `select_action(state)`: Sample action from policy
- `evaluate(state, action)`: Get log prob and value
- `update(buffer)`: PPO update with clipping
- `save()` / `load()`: Model checkpointing

**Hyperparameters**:

- Learning rate: 3e-4 (actor), 3e-4 (critic)
- Gamma (discount): 0.99
- Lambda (GAE): 0.95
- Clipping epsilon: 0.2
- Entropy coefficient: 0.01
- Value loss coefficient: 0.5
- Max gradient norm: 0.5
- Update epochs: 10
- Batch size: 64

### 4. Training Script (`train.py`)

**Purpose**: Main training loop

**Features**:

- Episode collection with multiple parallel environments (optional)
- Experience buffer management
- Periodic evaluation
- Checkpointing (save best model)
- Logging (TensorBoard or CSV)
- Progress visualization

**Training Loop**:

1. Collect trajectories (N steps)
2. Compute advantages using GAE
3. Update policy/value networks (K epochs)
4. Evaluate periodically
5. Save checkpoints

### 5. Evaluation Script (`evaluate.py`)

**Purpose**: Test trained agent

**Features**:

- Load trained model
- Run evaluation episodes
- Compare with baseline (no yaw control, greedy control)
- Visualize yaw angle trajectories
- Plot power output over time
- Generate performance metrics

### 6. Configuration (`config.py`)

**Purpose**: Centralized hyperparameters and settings

**Includes**:

- Wind farm configuration:
  - Layout name (e.g., `'sowfa_9turb_apc_alm_turbl'`)
  - WFSim path (configurable)
  - Number of turbines (extracted from layout)
- PPO hyperparameters
- Environment parameters (episode length, reward scaling)
- MATLAB paths and WFSim settings:
  - Path to WFSim directory
  - Model options (via `solverSet_default()`)
  - Simulation timestep (`Wp.sim.h`)
- Training parameters (episodes, evaluation frequency)

### 7. Utilities (`utils.py`)

**Purpose**: Helper functions

**Includes**:

- Reward normalization/scaling
- State normalization
- Visualization functions
- Logging utilities
- Data saving/loading

## File Structure

```
RL-for-Wind-Farm/
├── test.py (existing, can be removed or repurposed)
├── config.py
├── matlab_interface.py
├── wind_farm_env.py
├── ppo_agent.py
├── train.py
├── evaluate.py
├── utils.py
├── requirements.txt
├── README.md
└── checkpoints/
    └── (saved models)
```

## Implementation Order

1. **MATLAB Interface** - Establish communication with WFSim
2. **Environment Wrapper** - Create Gymnasium environment
3. **PPO Agent** - Implement PPO algorithm
4. **Training Script** - Build training loop
5. **Evaluation Script** - Test and visualize results
6. **Configuration & Utils** - Polish and organize

## Key Considerations

**MATLAB Integration**:

- Ensure MATLAB Engine API is installed: `pip install matlabengine`
- MATLAB must be on system PATH
- WFSim directory path must be provided (configurable in `config.py`)
- MATLAB engine lifecycle:
  - Start engine once at initialization
  - Add WFSim paths: `layoutDefinitions`, `controlDefinitions`, `solverDefinitions`
  - Call `WFSim_addpaths.m` to add essential paths
  - Keep engine alive for entire training session (don't restart each episode)
  - Close engine on environment cleanup
- Handle MATLAB data types:
  - Convert Python lists/arrays to MATLAB arrays using `matlab.double()`
  - Extract MATLAB arrays back to numpy arrays
  - Handle struct conversions (Wp, sol, sys, turbInput)

**State Space Design**:

- Normalize inputs for neural network stability
- Include sufficient information for wake steering decisions
- Balance detail vs. computational cost

**Action Space**:

- Convert degrees to radians if WFSim expects radians
- Implement action clipping to enforce constraints
- Consider action smoothing to avoid rapid changes

**Reward Shaping**:

- Normalize power output for stable learning
- Consider relative improvement vs. absolute power
- Add penalties for constraint violations

**Training Stability**:

- Use gradient clipping
- Monitor value function estimates
- Implement early stopping if needed
- Save multiple checkpoints

## Dependencies

- `numpy` - Numerical operations
- `torch` - Deep learning framework
- `gymnasium` - RL environment interface
- `matlab.engine` - MATLAB integration
- `matplotlib` - Visualization
- `tensorboard` (optional) - Training visualization
