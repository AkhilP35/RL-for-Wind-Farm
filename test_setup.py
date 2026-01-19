"""
Test script to verify WFSim and RL setup
Run this before training to ensure everything is configured correctly
"""

import sys
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required packages are installed"""
    logger.info("Testing Python package imports...")
    
    try:
        import numpy
        logger.info(f"✓ numpy {numpy.__version__}")
    except ImportError as e:
        logger.error(f"✗ numpy not found: {e}")
        return False
    
    try:
        import torch
        logger.info(f"✓ torch {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ torch not found: {e}")
        return False
    
    try:
        import gymnasium
        logger.info(f"✓ gymnasium {gymnasium.__version__}")
    except ImportError as e:
        logger.error(f"✗ gymnasium not found: {e}")
        return False
    
    try:
        import matplotlib
        logger.info(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        logger.error(f"✗ matplotlib not found: {e}")
        return False
    
    try:
        import matlab.engine
        logger.info("✓ matlab.engine")
    except ImportError as e:
        logger.error(f"✗ matlab.engine not found: {e}")
        logger.error("  Install MATLAB Engine API: cd matlabroot/extern/engines/python && python setup.py install")
        return False
    
    return True

def test_matlab_engine():
    """Test MATLAB engine connection"""
    logger.info("\nTesting MATLAB engine connection...")
    
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        result = eng.sqrt(16.0)
        logger.info(f"✓ MATLAB engine working (sqrt(16) = {result})")
        eng.quit()
        return True
    except Exception as e:
        logger.error(f"✗ MATLAB engine failed: {e}")
        logger.error("  Make sure MATLAB is installed and on your PATH")
        return False

def test_wfsim_path():
    """Test WFSim path configuration"""
    logger.info("\nTesting WFSim path...")
    
    try:
        import config
        from pathlib import Path
        
        wfsim_path = config.get_wfsim_path()
        logger.info(f"✓ WFSim path: {wfsim_path}")
        
        # Check if path exists
        if not wfsim_path.exists():
            logger.error(f"✗ WFSim path does not exist: {wfsim_path}")
            return False
        
        # Check for key files/directories
        required_items = [
            'layoutDefinitions',
            'controlDefinitions',
            'solverDefinitions',
            'WFSim_addpaths.m'
        ]
        
        missing = []
        for item in required_items:
            item_path = wfsim_path / item
            if not item_path.exists():
                missing.append(item)
        
        if missing:
            logger.error(f"✗ Missing WFSim components: {missing}")
            return False
        
        logger.info("✓ WFSim directory structure looks good")
        return True
        
    except Exception as e:
        logger.error(f"✗ WFSim path test failed: {e}")
        return False

def test_matlab_interface():
    """Test MATLAB interface with WFSim"""
    logger.info("\nTesting MATLAB interface with WFSim...")
    
    try:
        from matlab_interface import WFSimInterface
        import config
        
        wfsim_path = config.get_wfsim_path()
        layout_name = config.LAYOUT_NAME
        
        logger.info(f"  Initializing WFSim with layout: {layout_name}")
        logger.info("  This may take a moment...")
        
        wfsim = WFSimInterface(str(wfsim_path), layout_name)
        
        # Test reset
        logger.info("  Testing reset...")
        state = wfsim.reset()
        logger.info(f"✓ Reset successful (n_turbines: {wfsim.get_n_turbines()})")
        
        # Test step
        logger.info("  Testing step (zero yaw angles)...")
        zero_yaw = np.zeros(wfsim.get_n_turbines())
        state, power, done, info = wfsim.step(zero_yaw)
        logger.info(f"✓ Step successful (power: {power:.2e} W)")
        
        # Test a few more steps
        logger.info("  Testing multiple steps...")
        for i in range(3):
            state, power, done, info = wfsim.step(zero_yaw)
        logger.info(f"✓ Multiple steps successful (final power: {power:.2e} W)")
        
        wfsim.close()
        logger.info("✓ MATLAB interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ MATLAB interface test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_environment():
    """Test Gymnasium environment"""
    logger.info("\nTesting Wind Farm Environment...")
    
    try:
        from wind_farm_env import WindFarmEnv
        import config
        
        wfsim_path = config.get_wfsim_path()
        layout_name = config.LAYOUT_NAME
        
        logger.info("  Creating environment...")
        env = WindFarmEnv(wfsim_path=str(wfsim_path), layout_name=layout_name)
        
        logger.info(f"  Observation space: {env.observation_space.shape}")
        logger.info(f"  Action space: {env.action_space.shape}")
        
        # Test reset
        logger.info("  Testing reset...")
        obs, info = env.reset()
        logger.info(f"✓ Reset successful (obs shape: {obs.shape})")
        
        # Test step
        logger.info("  Testing step...")
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, step_info = env.step(action)
        logger.info(f"✓ Step successful (reward: {reward:.2f}, power: {step_info['power']:.2e} W)")
        
        # Test a few more steps
        logger.info("  Testing multiple steps...")
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
        logger.info(f"✓ Multiple steps successful")
        
        env.close()
        logger.info("✓ Environment test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_ppo_agent():
    """Test PPO agent creation"""
    logger.info("\nTesting PPO Agent...")
    
    try:
        from ppo_agent import PPOAgent
        
        # Create agent with small state/action dims for testing
        state_dim = 20
        action_dim = 3
        
        logger.info(f"  Creating agent (state_dim={state_dim}, action_dim={action_dim})...")
        agent = PPOAgent(state_dim, action_dim)
        logger.info("✓ Agent created successfully")
        
        # Test action selection
        logger.info("  Testing action selection...")
        state = np.random.randn(state_dim)
        action, log_prob, value = agent.select_action(state)
        logger.info(f"✓ Action selection successful (action shape: {action.shape}, value: {value:.2f})")
        
        logger.info("✓ PPO agent test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ PPO agent test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("WFSim RL Setup Test")
    logger.info("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("MATLAB Engine", test_matlab_engine),
        ("WFSim Path", test_wfsim_path),
        ("MATLAB Interface", test_matlab_interface),
        ("Environment", test_environment),
        ("PPO Agent", test_ppo_agent),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            logger.info("\n\nTests interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! You're ready to start training.")
        logger.info("Run: python train.py --episodes 100")
    else:
        logger.warning("\n⚠️  Some tests failed. Please fix the issues before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()
