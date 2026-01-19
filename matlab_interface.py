"""
MATLAB Interface for WFSim
Handles communication between Python and MATLAB WFSim simulator
"""

import numpy as np
import matlab.engine
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WFSimInterface:
    """
    Interface to MATLAB WFSim simulator
    
    Handles:
    - MATLAB engine lifecycle
    - WFSim initialization
    - Simulation stepping
    - State extraction
    """
    
    def __init__(self, wfsim_path: str, layout_name: str, model_options: str = "solverSet_default"):
        """
        Initialize MATLAB interface and WFSim
        
        Args:
            wfsim_path: Path to WFSim directory
            layout_name: Name of layout function (e.g., 'sowfa_9turb_apc_alm_turbl')
            model_options: Name of solver options function (default: 'solverSet_default')
        """
        self.wfsim_path = wfsim_path
        self.layout_name = layout_name
        self.model_options = model_options
        
        # MATLAB engine and WFSim structs
        self.eng = None
        self.Wp = None
        self.sol = None
        self.sys = None
        self.model_options_struct = None
        
        # Turbine information
        self.n_turbines = None
        self.baseline_CT_prime = None
        
        # Initialize MATLAB engine
        self._start_matlab_engine()
        self._initialize_wfsim()
        
    def _start_matlab_engine(self):
        """Start MATLAB engine and add WFSim paths"""
        logger.info("Starting MATLAB engine...")
        try:
            self.eng = matlab.engine.start_matlab()
            logger.info("MATLAB engine started successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to start MATLAB engine: {e}")
        
        # Add WFSim paths
        wfsim_path_str = str(self.wfsim_path)
        self.eng.addpath(wfsim_path_str, nargout=0)
        self.eng.addpath(self.eng.fullfile(wfsim_path_str, 'layoutDefinitions'), nargout=0)
        self.eng.addpath(self.eng.fullfile(wfsim_path_str, 'controlDefinitions'), nargout=0)
        self.eng.addpath(self.eng.fullfile(wfsim_path_str, 'solverDefinitions'), nargout=0)
        
        # Run WFSim_addpaths.m if it exists
        addpaths_file = self.eng.fullfile(wfsim_path_str, 'WFSim_addpaths.m')
        if self.eng.exist(addpaths_file, 'file'):
            self.eng.run(self.eng.fullfile(wfsim_path_str, 'WFSim_addpaths.m'), nargout=0)
        
        logger.info("WFSim paths added to MATLAB")
    
    def _initialize_wfsim(self):
        """Initialize WFSim with selected layout"""
        logger.info(f"Initializing WFSim with layout: {self.layout_name}")
        
        try:
            # Load layout
            layout_func = getattr(self.eng, f'layoutSet_{self.layout_name}')
            self.Wp = layout_func()
            
            # Get number of turbines
            # MATLAB struct access: Wp.turbine.N
            turbine_struct = self.Wp['turbine']
            if isinstance(turbine_struct, dict):
                self.n_turbines = int(turbine_struct['N'])
            else:
                # Alternative access method
                self.n_turbines = int(self.eng.getfield(self.Wp, 'turbine', 'N'))
            logger.info(f"Wind farm has {self.n_turbines} turbines")
            
            # Setup model options
            solver_func = getattr(self.eng, self.model_options)
            self.model_options_struct = solver_func(self.Wp)
            
            # Initialize WFSim
            # Note: InitWFSim returns (Wp, sol, sys) but may modify Wp in place
            result = self.eng.InitWFSim(self.Wp, self.model_options_struct, matlab.double([0.0]))
            if isinstance(result, tuple) and len(result) >= 3:
                self.Wp, self.sol, self.sys = result
            else:
                # If InitWFSim modifies Wp in place, we may need to handle differently
                # Try alternative approach
                self.eng.eval(f'[Wp, sol, sys] = InitWFSim(Wp, {self.model_options}, 0);', nargout=0)
                self.Wp = self.eng.workspace['Wp']
                self.sol = self.eng.workspace['sol']
                self.sys = self.eng.workspace['sys']
            
            # Get baseline CT_prime values (for reference, we'll keep these constant)
            self._get_baseline_CT_prime()
            
            logger.info("WFSim initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WFSim: {e}")
            raise
    
    def _get_baseline_CT_prime(self):
        """Get baseline CT_prime values from control set or compute default"""
        try:
            # Try to load control set to get baseline CT_prime
            control_func_name = f'controlSet_{self.layout_name}'
            if hasattr(self.eng, control_func_name):
                control_func = getattr(self.eng, control_func_name)
                turb_input_set = control_func(self.Wp)
                # Get first timestep CT_prime values as baseline
                if 'CT_prime' in turb_input_set:
                    ct_prime_array = turb_input_set['CT_prime']
                    # Extract first column (first timestep)
                    self.baseline_CT_prime = [
                        float(ct_prime_array[i][0]) for i in range(self.n_turbines)
                    ]
                else:
                    # Default CT_prime if not available
                    self.baseline_CT_prime = [0.8] * self.n_turbines
            else:
                # Default CT_prime
                self.baseline_CT_prime = [0.8] * self.n_turbines
        except Exception as e:
            logger.warning(f"Could not get baseline CT_prime, using default: {e}")
            self.baseline_CT_prime = [0.8] * self.n_turbines
    
    def reset(self) -> Dict:
        """
        Reset WFSim to initial state
        
        Returns:
            Dictionary containing initial state information
        """
        logger.debug("Resetting WFSim simulation")
        
        # Reinitialize WFSim
        try:
            result = self.eng.InitWFSim(self.Wp, self.model_options_struct, matlab.double([0.0]))
            if isinstance(result, tuple) and len(result) >= 3:
                self.Wp, self.sol, self.sys = result
            else:
                self.eng.eval('[Wp, sol, sys] = InitWFSim(Wp, modelOptions, 0);', nargout=0)
                self.Wp = self.eng.workspace['Wp']
                self.sol = self.eng.workspace['sol']
                self.sys = self.eng.workspace['sys']
        except Exception as e:
            logger.error(f"Failed to reset WFSim: {e}")
            raise
        
        return self.get_state()
    
    def step(self, yaw_angles: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Advance simulation one timestep with given yaw angles
        
        Args:
            yaw_angles: Array of yaw angles in degrees, shape (n_turbines,)
            
        Returns:
            Tuple of (state_dict, reward, done, info)
        """
        if len(yaw_angles) != self.n_turbines:
            raise ValueError(f"Expected {self.n_turbines} yaw angles, got {len(yaw_angles)}")
        
        # Clip yaw angles to valid range
        yaw_angles = np.clip(yaw_angles, -30.0, 30.0)
        
        # Create turbInput struct
        turb_input = self.eng.struct()
        turb_input['t'] = matlab.double([self.sol['time']])
        
        # Set yaw angles (phi)
        turb_input['phi'] = matlab.double(yaw_angles.tolist())
        
        # Set CT_prime to baseline values
        turb_input['CT_prime'] = matlab.double(self.baseline_CT_prime)
        
        # Advance simulation
        try:
            result = self.eng.WFSim_timestepping(
                self.sol, self.sys, self.Wp, turb_input, self.model_options_struct
            )
            if isinstance(result, tuple) and len(result) >= 2:
                self.sol, self.sys = result
            else:
                # Alternative approach if function modifies in place
                self.eng.eval('[sol, sys] = WFSim_timestepping(sol, sys, Wp, turbInput, modelOptions);', nargout=0)
                self.sol = self.eng.workspace['sol']
                self.sys = self.eng.workspace['sys']
        except Exception as e:
            logger.error(f"WFSim timestepping failed: {e}")
            raise
        
        # Extract state and compute reward
        state = self.get_state()
        reward = self.get_total_power()
        done = False  # Episode termination handled by environment
        info = {
            'power_per_turbine': self.get_power_per_turbine(),
            'yaw_angles': yaw_angles.copy(),
        }
        
        return state, reward, done, info
    
    def get_state(self) -> Dict:
        """
        Extract state from current sol struct
        
        Returns:
            Dictionary containing state information
        """
        state = {}
        
        # Extract flow fields (u, v)
        sol_u = np.array(self.sol['u'])
        sol_v = np.array(self.sol['v'])
        state['u'] = sol_u
        state['v'] = sol_v
        
        # Extract turbine outputs
        try:
            if 'turbine' in self.sol:
                turbine = self.sol['turbine']
                if isinstance(turbine, dict):
                    state['turbine_power'] = np.array(turbine.get('power', [0]*self.n_turbines)).flatten()
                    state['turbine_phi'] = np.array(turbine.get('phi', [0]*self.n_turbines)).flatten()
                    if 'CT' in turbine:
                        state['turbine_CT'] = np.array(turbine['CT']).flatten()
                else:
                    # Use MATLAB engine to access
                    state['turbine_power'] = np.array(self.eng.getfield(self.sol, 'turbine', 'power')).flatten()
                    state['turbine_phi'] = np.array(self.eng.getfield(self.sol, 'turbine', 'phi')).flatten()
                    try:
                        state['turbine_CT'] = np.array(self.eng.getfield(self.sol, 'turbine', 'CT')).flatten()
                    except:
                        pass
            else:
                # Fallback if turbine struct not available
                state['turbine_power'] = np.zeros(self.n_turbines)
                state['turbine_phi'] = np.zeros(self.n_turbines)
        except Exception as e:
            logger.warning(f"Could not extract turbine outputs: {e}")
            state['turbine_power'] = np.zeros(self.n_turbines)
            state['turbine_phi'] = np.zeros(self.n_turbines)
        
        # Extract time information
        state['time'] = float(self.sol['time'])
        state['k'] = int(self.sol['k'])
        
        # Extract turbine locations for flow field sampling
        try:
            turbine_struct = self.Wp['turbine']
            if isinstance(turbine_struct, dict):
                crx = turbine_struct.get('Crx', [])
                cry = turbine_struct.get('Cry', [])
            else:
                # Use MATLAB engine to access struct fields
                crx = self.eng.getfield(self.Wp, 'turbine', 'Crx')
                cry = self.eng.getfield(self.Wp, 'turbine', 'Cry')
            
            turbine_locs = []
            for i in range(self.n_turbines):
                # Turbine locations in grid coordinates
                if isinstance(crx, (list, np.ndarray)):
                    idx = int(crx[i]) - 1  # Convert to 0-indexed
                    idy = int(cry[i]) - 1
                else:
                    # MATLAB array access
                    idx = int(self.eng.workspace['Wp']['turbine']['Crx'][i]) - 1
                    idy = int(self.eng.workspace['Wp']['turbine']['Cry'][i]) - 1
                turbine_locs.append((idx, idy))
            state['turbine_locations'] = turbine_locs
        except Exception as e:
            logger.warning(f"Could not extract turbine locations: {e}")
            state['turbine_locations'] = []
        
        return state
    
    def get_power_per_turbine(self) -> np.ndarray:
        """Get power output for each turbine"""
        state = self.get_state()
        return state['turbine_power']
    
    def get_total_power(self) -> float:
        """Get total power output of all turbines"""
        power_per_turbine = self.get_power_per_turbine()
        return float(np.sum(power_per_turbine))
    
    def get_n_turbines(self) -> int:
        """Get number of turbines"""
        return self.n_turbines
    
    def close(self):
        """Close MATLAB engine"""
        if self.eng is not None:
            logger.info("Closing MATLAB engine")
            try:
                self.eng.quit()
            except Exception as e:
                logger.warning(f"Error closing MATLAB engine: {e}")
            self.eng = None
