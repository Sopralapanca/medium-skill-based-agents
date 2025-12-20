# Test installation
def test_installation():
    """Test if all required packages are properly installed"""
    print("Testing installation...")

    try:
        import gymnasium as gym
        print("✓ Gymnasium imported successfully")
    except ImportError as e:
        print(f"✗ Gymnasium import failed: {e}")
        return False

    try:
        import ale_py
        gym.register_envs(ale_py)
        print("✓ ALE-Py imported and registered successfully")
    except ImportError as e:
        print(f"✗ ALE-Py import failed: {e}")
        return False

    try:
        from stable_baselines3 import PPO
        print("✓ Stable-Baselines3 imported successfully")
    except ImportError as e:
        print(f"✗ Stable-Baselines3 import failed: {e}")
        return False

    # Test environment availability
    try:
        envs = list(gym.envs.registry.keys())
        print(envs)
    except Exception as e:
        print(f"⚠ Could not check environments: {e}")

    return True

# Run tests
if __name__ == "__main__":
    all_tests_passed = test_installation()
    if all_tests_passed:
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Please check the output above.")