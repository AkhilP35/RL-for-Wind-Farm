"""
Wind Farm Environment for Reinforcement Learning
Gymnasium-compatible environment wrapping WFSim
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging

from matlab_interface import WFSimInterface
import config

logger = logging.getLogger(__name__)


class WindFarmEnv(gym.Env):
    """
    Wind Farm Control Environment
    
    Controls turbine yaw angles to maximize total power output through wake steering.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        wfsim_path: Optional[str] = None,
        layout_name: Optional[str] = None,
        episode_length: Optional[int] = None,
        reward_scale: Optional[float] = None,
        use_normalized_rewards: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize Wind Farm Environment
        
        Args:
            wfsim_path: Path to WFSim directory (uses config if None)
            layout_name: Layout name (uses config if None)
            episode_length: Episode length in timesteps (uses config if None)
            reward_scale: Reward scaling factor (uses config if None)
            use_normalized_rewards: Whether to normalize rewards (uses config if None)
        """
        super().__init__()
        
        # Configuration
        self.wfsim_path = wfsim_path or config.get_wfsim_path()
        self.layout_name = layout_name or config.LAYOUT_NAME
        self.episode_length = episode_length or config.EPISODE_LENGTH
        self.reward_scale = reward_scale or config.REWARD_SCALE
        self.use_normalized_rewards = use_normalized_rewards or config.USE_NORMALIZED_REWARDS
        
        # Initialize MATLAB interface
        self.wfsim = WFSimInterface(
            str(self.wfsim_path),
            self.layout_name,
            config.MODEL_OPTIONS
        )
        self.n_turbines = self.wfsim.get_n_turbines()
        
        # Episode tracking
        self.current_step = 0
        self.baseline_power = None  # Baseline power for normalization
        self.episode_powers = []  # Track power over episode
        
        # Define action space: yaw angles for each turbine
        self.action_space = spaces.Box(
            low=config.YAW_MIN,
            high=config.YAW_MAX,
            shape=(self.n_turbines,),
            dtype=np.float32
        )
        
        # Define observation space
        obs_dim = self._compute_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        logger.info(f"WindFarmEnv initialized: {self.n_turbines} turbines, obs_dim={obs_dim}")
    
    def _compute_observation_dim(self) -> int:
        """Compute observation space dimension based on config"""
        dim = 0
        
        # Flow velocities at turbine locations (u, v)
        if config.INCLUDE_FLOW_FIELD:
            dim += 2 * self.n_turbines
        
        # Turbine power outputs
        if config.INCLUDE_TURBINE_POWER:
            dim += self.n_turbines
        
        # Current yaw angles
        if config.INCLUDE_CURRENT_YAW:
            dim += self.n_turbines
        
        # Thrust coefficients
        if config.INCLUDE_THRUST_COEFF:
            dim += self.n_turbines
        
        # Flow field regions around turbines
        if config.INCLUDE_FLOW_FIELD and config.FLOW_FIELD_REGION_SIZE > 0:
            region_size = config.FLOW_FIELD_REGION_SIZE
            dim += 2 * self.n_turbines * (region_size ** 2)  # u and v for each point
        
        return dim
    
    def _extract_observation(self, state: Dict) -> np.ndarray:
        """
        Extract observation vector from state dictionary
        
        Args:
            state: State dictionary from WFSim
            
        Returns:
            Observation vector as numpy array
        """
        obs_parts = []
        
        # Get flow fields
        u_field = state['u']
        v_field = state['v']
        turbine_locs = state.get('turbine_locations', [])
        
        # Flow velocities at turbine locations
        if config.INCLUDE_FLOW_FIELD:
            u_at_turbines = []
            v_at_turbines = []
            for idx, idy in turbine_locs:
                if 0 <= idx < u_field.shape[0] and 0 <= idy < u_field.shape[1]:
                    u_at_turbines.append(float(u_field[idx, idy]))
                    v_at_turbines.append(float(v_field[idx, idy]))
                else:
                    u_at_turbines.append(0.0)
                    v_at_turbines.append(0.0)
            obs_parts.extend(u_at_turbines)
            obs_parts.extend(v_at_turbines)
        
        # Turbine power outputs (normalized)
        if config.INCLUDE_TURBINE_POWER:
            power = state['turbine_power']
            # Normalize by a reference power (e.g., rated power or max observed)
            if self.baseline_power is not None and self.baseline_power > 0:
                power_normalized = power / (self.baseline_power / self.n_turbines)
            else:
                power_normalized = power / 1e6  # Rough normalization by MW
            obs_parts.extend(power_normalized.tolist())
        
        # Current yaw angles (normalized to [-1, 1])
        if config.INCLUDE_CURRENT_YAW:
            yaw = state['turbine_phi']
            yaw_normalized = yaw / config.YAW_MAX  # Normalize to [-1, 1]
            obs_parts.extend(yaw_normalized.tolist())
        
        # Thrust coefficients
        if config.INCLUDE_THRUST_COEFF and 'turbine_CT' in state:
            ct = state['turbine_CT']
            obs_parts.extend(ct.tolist())
        
        # Flow field regions around turbines
        if config.INCLUDE_FLOW_FIELD and config.FLOW_FIELD_REGION_SIZE > 0:
            region_size = config.FLOW_FIELD_REGION_SIZE
            half_size = region_size // 2
            
            for idx, idy in turbine_locs:
                # Extract region around turbine
                u_region = []
                v_region = []
                for di in range(-half_size, half_size + 1):
                    for dj in range(-half_size, half_size + 1):
                        i, j = idx + di, idy + dj
                        if 0 <= i < u_field.shape[0] and 0 <= j < u_field.shape[1]:
                            u_region.append(float(u_field[i, j]))
                            v_region.append(float(v_field[i, j]))
                        else:
                            u_region.append(0.0)
                            v_region.append(0.0)
                obs_parts.extend(u_region)
                obs_parts.extend(v_region)
        
        return np.array(obs_parts, dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset WFSim
        state = self.wfsim.reset()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_powers = []
        
        # Compute baseline power (power with zero yaw angles)
        # This is used for reward normalization
        if self.use_normalized_rewards:
            zero_yaw = np.zeros(self.n_turbines)
            _, baseline_reward, _, _ = self.wfsim.step(zero_yaw)
            self.baseline_power = baseline_reward
            # Reset again after baseline measurement
            state = self.wfsim.reset()
        
        # Extract observation
        observation = self._extract_observation(state)
        
        info = {
            'state': state,
            'baseline_power': self.baseline_power,
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep
        
        Args:
            action: Yaw angles for each turbine (degrees)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip action to valid range
        action = np.clip(action, config.YAW_MIN, config.YAW_MAX)
        
        # Advance simulation
        state, power, done, step_info = self.wfsim.step(action)
        
        # Compute reward
        if self.use_normalized_rewards and self.baseline_power is not None:
            # Normalized reward: improvement over baseline
            reward = (power - self.baseline_power) / self.baseline_power
        else:
            # Raw power output (scaled)
            reward = power * self.reward_scale
        
        # Track episode power
        self.episode_powers.append(power)
        self.current_step += 1
        
        # Check termination
        terminated = False  # No natural termination condition
        truncated = self.current_step >= self.episode_length
        
        # Extract observation
        observation = self._extract_observation(state)
        
        info = {
            'state': state,
            'power': power,
            'power_per_turbine': step_info['power_per_turbine'],
            'yaw_angles': step_info['yaw_angles'],
            'episode_power_history': self.episode_powers.copy(),
            'step': self.current_step,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (placeholder for future visualization)"""
        # TODO: Implement visualization using WFSim animation functions
        pass
    
    def close(self):
        """Close environment and cleanup"""
        if hasattr(self, 'wfsim'):
            self.wfsim.close()
