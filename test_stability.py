#!/usr/bin/env python3

import mani_skill
import gymnasium as gym
import numpy as np
import torch

def test_robot_stability():
    """Test the stability of the xarm6_inspire_hand_right robot"""
    print("Testing xarm6_inspire_hand_right stability...")

    # Create environment
    env = gym.make('PickCube-v1', robot_uids='xarm6_inspire_hand_right', render_mode='rgb_array', num_envs=1, max_episode_steps=500)
    print(f"✓ Environment created with control mode: {env.unwrapped.control_mode}")
    print(f"✓ Action space shape: {env.action_space.shape}")

    # Reset environment
    obs, info = env.reset(seed=42)
    print("✓ Environment reset successfully")

    # Check initial joint positions
    qpos = env.unwrapped.agent.robot.get_qpos()
    print(f"Initial joint positions: {qpos[:5].cpu().numpy()}")  # Show first 5 joints

    # Check if controller has targets
    controller = env.unwrapped.agent.controllers[env.unwrapped.control_mode]
    print(f"Controller type: {type(controller)}")
    if hasattr(controller, 'target_qpos'):
        print(f"Controller target positions: {controller.target_qpos[:5].cpu().numpy()}")
    elif hasattr(controller, 'target'):
        print(f"Controller target: {controller.target[:5].cpu().numpy()}")
    else:
        print("Controller has no explicit target")

    # Check controller attributes
    print(f"Controller attributes: {[attr for attr in dir(controller) if not attr.startswith('_')]}")

    # Check if this is a joint position controller
    if hasattr(controller, 'p_gain'):
        print(f"P gain: {controller.p_gain}")
    if hasattr(controller, 'd_gain'):
        print(f"D gain: {controller.d_gain}")

    # Test for longer duration with very small actions
    stable_steps = 0
    max_steps = 1000  # Test for much longer duration to catch delayed instability

    for step in range(max_steps):
        # Use even smaller random actions
        action = np.random.uniform(-0.005, 0.005, size=env.action_space.shape)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Monitor joint limits
        current_qpos = env.unwrapped.agent.robot.get_qpos()
        joint_limits = env.unwrapped.agent.controllers[env.unwrapped.control_mode].single_action_space

        obs, reward, terminated, truncated, info = env.step(action)

        # Check if joints are moving reasonably (not exploding)
        qvel = env.unwrapped.agent.robot.get_qvel()
        max_joint_vel = torch.max(torch.abs(qvel)).item()

        # Get joint positions after action
        new_qpos = env.unwrapped.agent.robot.get_qpos()

        if max_joint_vel < 1000.0:  # Reasonable velocity threshold for small actions
            stable_steps += 1
        else:
            print(f"✗ Unstable at step {step}: max joint velocity = {max_joint_vel}")
            print(f"Action taken: {action}")
            print(f"Joint velocities: {qvel}")
            break

        # Check joint limits and print status
        if step % 20 == 0:  # Check every 20 steps
            print(f"Step {step}: max joint velocity = {max_joint_vel:.3f}, stable steps = {stable_steps}")
            print(f"Joint positions range: [{new_qpos.min().item():.3f}, {new_qpos.max().item():.3f}]")
            print(f"Action range: [{action.min():.6f}, {action.max():.6f}]")

        if terminated or truncated:
            print(f"Episode ended at step {step}, terminated={terminated}, truncated={truncated}")
            print(f"Final joint positions: {new_qpos}")
            break

    env.close()

    stability_ratio = stable_steps / max_steps
    print(".2%")

    if stability_ratio > 0.8:
        print("✅ Robot appears stable!")
        return True
    else:
        print("❌ Robot may be unstable")
        return False

if __name__ == "__main__":
    test_robot_stability()
