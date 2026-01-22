"""
MATLAB Interface for WFSim
Handles communication between Python and MATLAB WFSim simulator
"""

import numpy as np
import matlab.engine
from typing import Dict, List, Optional, Tuple
import logging
import json
import time
import sys
import os

logger = logging.getLogger(__name__)

# #region agent log
_DEBUG_LOG_PATH = "/Users/akhilpatel/Desktop/Dissertation/.cursor/debug.log"

def _dbg(hypothesis_id: str, location: str, message: str, data: Dict):
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
        payload = {
            "sessionId": "debug-session",
            "runId": "run2",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion agent log

# #region agent log
_dbg(
    "D",
    "matlab_interface.py:module_import",
    "matlab_interface module imported",
    {"python_exe": sys.executable, "python_version": sys.version.split(" ")[0]},
)
# #endregion agent log


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
        # NOTE: WFSim `sys` contains sparse matrices and cannot be converted to Python
        # via matlab.engine. We keep `sys` inside the MATLAB workspace only.
        self.sys = None
        self._sys_in_matlab_workspace = False
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

        # #region agent log
        _dbg(
            "A",
            "matlab_interface.py:_initialize_wfsim:entry",
            "Entering WFSim init",
            {
                "layout_name": self.layout_name,
                "model_options": self.model_options,
                "python_exe": sys.executable,
                "python_version": sys.version.split(" ")[0],
            },
        )
        # #endregion agent log

        try:
            # Load layout
            layout_func = getattr(self.eng, f'layoutSet_{self.layout_name}')
            self.Wp = layout_func()
            
            # Get number of turbines
            # Number of turbines is determined by length of Crx or Cry array
            # Use MATLAB engine to access struct fields properly
            self.eng.workspace['Wp'] = self.Wp
            # Get length of Crx array (number of turbines)
            crx_length = self.eng.eval('length(Wp.turbine.Crx)', nargout=1)
            self.n_turbines = int(crx_length)
            logger.info(f"Wind farm has {self.n_turbines} turbines")
            
            # Setup model options
            solver_func = getattr(self.eng, self.model_options)
            self.model_options_struct = solver_func(self.Wp)
            
            # Initialize WFSim
            # Use MATLAB workspace to handle structs properly
            self.eng.workspace['Wp'] = self.Wp
            self.eng.workspace['modelOptions'] = self.model_options_struct
            self.eng.eval('[Wp, sol, sys] = InitWFSim(Wp, modelOptions, 0);', nargout=0)

            # #region agent log
            try:
                whos_sol = self.eng.eval("evalc('whos sol')", nargout=1)
            except Exception as e:
                whos_sol = f"ERROR:{type(e).__name__}:{e}"
            try:
                whos_sys = self.eng.eval("evalc('whos sys')", nargout=1)
            except Exception as e:
                whos_sys = f"ERROR:{type(e).__name__}:{e}"
            try:
                sys_class = self.eng.eval("class(sys)", nargout=1)
            except Exception as e:
                sys_class = f"ERROR:{type(e).__name__}:{e}"
            try:
                sys_fields = self.eng.eval("fieldnames(sys)", nargout=1)
                sys_fields = [str(x) for x in sys_fields]
            except Exception as e:
                sys_fields = [f"ERROR:{type(e).__name__}:{e}"]
            _dbg(
                "A",
                "matlab_interface.py:_initialize_wfsim:post_InitWFSim",
                "MATLAB variables after InitWFSim",
                {"whos_sol": str(whos_sol), "whos_sys": str(whos_sys), "sys_class": str(sys_class), "sys_fields": sys_fields},
            )
            # #endregion agent log

            self.Wp = self.eng.workspace['Wp']
            self.sol = self.eng.workspace['sol']
            # IMPORTANT: do not fetch `sys` into Python (it contains sparse arrays).
            self._sys_in_matlab_workspace = True
            # #region agent log
            _dbg(
                "C",
                "matlab_interface.py:_initialize_wfsim:sys_workspace_only",
                "Keeping sys in MATLAB workspace only (no Python conversion)",
                {"sys_fields": sys_fields},
            )
            # #endregion agent log
            
            # Get baseline CT_prime values (for reference, we'll keep these constant)
            self._get_baseline_CT_prime()
            
            logger.info("WFSim initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WFSim: {e}")
            # #region agent log
            _dbg(
                "B",
                "matlab_interface.py:_initialize_wfsim:exception",
                "WFSim init raised exception",
                {"error_type": type(e).__name__, "error": str(e)},
            )
            # #endregion agent log
            raise
    
    def _get_baseline_CT_prime(self):
        """Get baseline CT_prime values from control set or compute default"""
        try:
            # Try to load control set to get baseline CT_prime
            control_func_name = f'controlSet_{self.layout_name}'
            if hasattr(self.eng, control_func_name):
                control_func = getattr(self.eng, control_func_name)
                self.eng.workspace['Wp'] = self.Wp
                turb_input_set = control_func(self.Wp)
                
                # Get first timestep CT_prime values as baseline using MATLAB eval
                self.eng.workspace['turbInputSet'] = turb_input_set
                ct_prime_array = np.array(self.eng.eval('turbInputSet.CT_prime(:,1)', nargout=1)).flatten()
                self.baseline_CT_prime = [float(ct_prime_array[i]) for i in range(self.n_turbines)]
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
            self.eng.workspace['Wp'] = self.Wp
            self.eng.workspace['modelOptions'] = self.model_options_struct
            self.eng.eval('[Wp, sol, sys] = InitWFSim(Wp, modelOptions, 0);', nargout=0)
            self.Wp = self.eng.workspace['Wp']
            self.sol = self.eng.workspace['sol']
            self._sys_in_matlab_workspace = True
            # #region agent log
            _dbg(
                "C",
                "matlab_interface.py:reset:sys_workspace_only",
                "Reset kept sys in MATLAB workspace only",
                {},
            )
            # #endregion agent log
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
        # Get current time (ensure sol is in workspace)
        self.eng.workspace['sol'] = self.sol
        current_time = float(self.eng.eval('sol.time', nargout=1))
        
        turb_input = self.eng.struct()
        turb_input['t'] = matlab.double([current_time])
        
        # Set yaw angles (phi)
        turb_input['phi'] = matlab.double(yaw_angles.tolist())
        
        # Set CT_prime to baseline values
        turb_input['CT_prime'] = matlab.double(self.baseline_CT_prime)
        
        # Advance simulation
        try:
            # Use MATLAB workspace to handle structs properly
            self.eng.workspace['sol'] = self.sol
            self.eng.workspace['Wp'] = self.Wp
            self.eng.workspace['turbInput'] = turb_input
            self.eng.workspace['modelOptions'] = self.model_options_struct
            # #region agent log
            _dbg(
                "C",
                "matlab_interface.py:step:pre_timestepping",
                "Calling WFSim_timestepping using MATLAB-workspace sys",
                {"sys_in_workspace": bool(self._sys_in_matlab_workspace), "yaw_min": float(np.min(yaw_angles)), "yaw_max": float(np.max(yaw_angles))},
            )
            # #endregion agent log
            self.eng.eval('[sol, sys] = WFSim_timestepping(sol, sys, Wp, turbInput, modelOptions);', nargout=0)
            self.sol = self.eng.workspace['sol']
            self._sys_in_matlab_workspace = True
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

        # #region agent log
        _dbg(
            "E",
            "matlab_interface.py:get_state:entry",
            "Entering get_state",
            {"sys_in_workspace": bool(getattr(self, "_sys_in_matlab_workspace", False)), "n_turbines": self.n_turbines},
        )
        # #endregion agent log
        
        # Extract flow fields (u, v)
        # Use MATLAB engine to access struct fields
        self.eng.workspace['sol'] = self.sol
        try:
            # #region agent log
            try:
                u_is_sparse = bool(self.eng.eval("issparse(sol.u)", nargout=1))
            except Exception as e:
                u_is_sparse = f"ERROR:{type(e).__name__}:{e}"
            try:
                v_is_sparse = bool(self.eng.eval("issparse(sol.v)", nargout=1))
            except Exception as e:
                v_is_sparse = f"ERROR:{type(e).__name__}:{e}"
            try:
                u_class = self.eng.eval("class(sol.u)", nargout=1)
            except Exception as e:
                u_class = f"ERROR:{type(e).__name__}:{e}"
            try:
                v_class = self.eng.eval("class(sol.v)", nargout=1)
            except Exception as e:
                v_class = f"ERROR:{type(e).__name__}:{e}"
            _dbg(
                "E",
                "matlab_interface.py:get_state:pre_uv_eval",
                "About to convert sol.u/sol.v to numpy",
                {"u_is_sparse": u_is_sparse, "v_is_sparse": v_is_sparse, "u_class": str(u_class), "v_class": str(v_class)},
            )
            # #endregion agent log

            sol_u = np.array(self.eng.eval('sol.u', nargout=1))
            sol_v = np.array(self.eng.eval('sol.v', nargout=1))
        except Exception as e:
            # #region agent log
            _dbg(
                "E",
                "matlab_interface.py:get_state:uv_eval_exception",
                "Failed converting sol.u/sol.v",
                {"error_type": type(e).__name__, "error": str(e)},
            )
            # #endregion agent log
            raise
        state['u'] = sol_u
        state['v'] = sol_v
        
        # Extract turbine outputs
        try:
            # Use MATLAB engine to access struct fields
            self.eng.workspace['sol'] = self.sol
            # #region agent log
            try:
                sol_fields = self.eng.eval("fieldnames(sol)", nargout=1)
                sol_fields = [str(x) for x in sol_fields]
            except Exception as e:
                sol_fields = [f"ERROR:{type(e).__name__}:{e}"]
            try:
                has_turbine = bool(self.eng.eval("isfield(sol,'turbine')", nargout=1))
            except Exception as e:
                has_turbine = f"ERROR:{type(e).__name__}:{e}"
            _dbg(
                "F",
                "matlab_interface.py:get_state:sol_fields",
                "sol fieldnames + turbine presence",
                {"has_turbine": has_turbine, "sol_fields": sol_fields},
            )
            # #endregion agent log

            if not has_turbine:
                state['turbine_power'] = np.zeros(self.n_turbines)
                state['turbine_phi'] = np.zeros(self.n_turbines)
            else:
                # #region agent log
                try:
                    turb_fields = self.eng.eval("fieldnames(sol.turbine)", nargout=1)
                    turb_fields = [str(x) for x in turb_fields]
                except Exception as e:
                    turb_fields = [f"ERROR:{type(e).__name__}:{e}"]
                _dbg(
                    "F",
                    "matlab_interface.py:get_state:turbine_fields",
                    "sol.turbine fieldnames",
                    {"turbine_fields": turb_fields},
                )
                # #endregion agent log

                # power
                state['turbine_power'] = np.array(self.eng.eval('sol.turbine.power', nargout=1)).flatten()

                # yaw angle: try a few common names across WFSim variants
                yaw_expr = (
                    "if isfield(sol.turbine,'phi'); y=sol.turbine.phi;"
                    "elseif isfield(sol.turbine,'yaw'); y=sol.turbine.yaw;"
                    "elseif isfield(sol.turbine,'yawAngle'); y=sol.turbine.yawAngle;"
                    "elseif isfield(sol.turbine,'Phi'); y=sol.turbine.Phi;"
                    "else; y=zeros(size(sol.turbine.power)); end"
                )
                state['turbine_phi'] = np.array(self.eng.eval(yaw_expr, nargout=1)).flatten()

                # optional thrust coefficient
                try:
                    ct_expr = (
                        "if isfield(sol.turbine,'CT'); c=sol.turbine.CT;"
                        "elseif isfield(sol.turbine,'Ct'); c=sol.turbine.Ct;"
                        "else; c=[]; end"
                    )
                    ct_val = np.array(self.eng.eval(ct_expr, nargout=1)).flatten()
                    if ct_val.size:
                        state['turbine_CT'] = ct_val
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not extract turbine outputs: {e}")
            # #region agent log
            _dbg(
                "E",
                "matlab_interface.py:get_state:turbine_eval_exception",
                "Failed converting sol.turbine.*",
                {"error_type": type(e).__name__, "error": str(e)},
            )
            # #endregion agent log
            state['turbine_power'] = np.zeros(self.n_turbines)
            state['turbine_phi'] = np.zeros(self.n_turbines)
        
        # Extract time information
        state['time'] = float(self.eng.eval('sol.time', nargout=1))
        state['k'] = int(self.eng.eval('sol.k', nargout=1))
        
        # Extract turbine locations for flow field sampling
        try:
            # Use MATLAB engine to access struct fields
            self.eng.workspace['Wp'] = self.Wp
            crx = np.array(self.eng.eval('Wp.turbine.Crx', nargout=1)).flatten()
            cry = np.array(self.eng.eval('Wp.turbine.Cry', nargout=1)).flatten()
            
            turbine_locs = []
            for i in range(self.n_turbines):
                # Turbine locations in grid coordinates (convert to 0-indexed)
                idx = int(crx[i]) - 1
                idy = int(cry[i]) - 1
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
