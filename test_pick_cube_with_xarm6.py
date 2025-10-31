# import gymnasium as gym
# import mani_skill.envs
#
# env = gym.make(
#     "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
#     robot_uids = "xarm6_inspire_hand_right",
#     num_envs=1,
#     obs_mode="state", # there is also "state_dict", "rgbd", ...
#     control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
#     render_mode="human",
#     sim_backend="cpu",
#     render_backend="cpu"
# )
# print("Observation space", env.observation_space)
# print("Action space", env.action_space)
#
# obs, _ = env.reset(seed=0) # reset with a seed for determinism
# done = False
#
#
#
#
# while True :
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     env.render()  # a display is required to
# env.close()
#
#



import time
import threading
import tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np
import gymnasium as gym
import mani_skill.envs


# ------------------------- GUI -------------------------
class ActionTuner:
    def __init__(self, dim: int, lows: np.ndarray, highs: np.ndarray, title: str = "Action Tuner"):
        assert dim == lows.shape[0] == highs.shape[0]
        self.dim = dim
        self.lows = lows
        self.highs = highs

        self.root = tk.Tk()
        self.root.title(title)

        # 关闭窗口时的标记
        self.is_alive = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 当前 action 数组（float），以及线程安全锁
        self._v = np.zeros(self.dim, dtype=np.float32)
        self._lock = threading.Lock()

        # Tk 变量 & 滑块
        self.vals = []
        for i in range(self.dim):
            # 将初始值置为动作空间中心
            mid = float((self.lows[i] + self.highs[i]) / 2.0)
            var = tk.DoubleVar(value=mid)
            self.vals.append(var)
            self._create_slider(f"Dim {i}", var, float(self.lows[i]), float(self.highs[i]), i)

        # 显示当前值
        self.label = tk.Label(self.root, text="")
        self.label.grid(row=self.dim, column=0, columnspan=2, pady=10, sticky="w")
        self._update_from_vars()  # 初始化一次
        for i in range(self.dim):
            self.vals[i].trace_add("write", self._on_var_change)

    def _on_close(self):
        self.is_alive = False
        self.root.destroy()

    def _create_slider(self, name, variable, min_val, max_val, row):
        label = tk.Label(self.root, text=f"{name} [{min_val:.2f}, {max_val:.2f}]")
        label.grid(row=row, column=0, padx=8, pady=4, sticky="e")

        slider = ttk.Scale(
            self.root, from_=min_val, to=max_val, orient="horizontal",
            variable=variable, length=600
        )
        slider.grid(row=row, column=1, padx=8, pady=4, sticky="w")

    def _on_var_change(self, *args):
        self._update_from_vars()

    def _update_from_vars(self):
        with self._lock:
            for i in range(self.dim):
                self._v[i] = float(self.vals[i].get())
            # 简短显示（避免太长）
            preview = np.array2string(self._v, precision=2, separator=", ", suppress_small=True)
            self.label.config(text=f"Action = {preview}")

    def get_action(self) -> np.ndarray:
        """线程安全读当前 action。"""
        with self._lock:
            return self._v.copy()

    def run(self):
        self.root.mainloop()


class ActionTunerRunner:
    """在后台线程启动 Tk GUI，主线程可随时 get_action。"""
    def __init__(self, dim: int, lows: np.ndarray, highs: np.ndarray, title: str = "Action Tuner"):
        self.app: Optional[ActionTuner] = None
        self._ready = threading.Event()

        def _worker():
            self.app = ActionTuner(dim, lows, highs, title)
            self._ready.set()
            self.app.run()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        # 等待 GUI 构建好
        self._ready.wait()

    def is_alive(self) -> bool:
        return self.app is not None and self.app.is_alive

    def get_action(self) -> np.ndarray:
        if self.app is None:
            return None
        return self.app.get_action()


# ------------------------- Env loop -------------------------
def main():
    # 1) 创建 ManiSkill 环境
    env = gym.make(
        "PickCube-v1",
        robot_uids="xarm6_inspire_hand_right",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="human",
        sim_backend="cpu",
        render_backend="cpu",
    )
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # 2) 从动作空间获取维度与上下界
    assert hasattr(env.action_space, "shape") and len(env.action_space.shape) == 1, \
        "只处理一维 Box 动作空间"
    act_dim = env.action_space.shape[0]
    lows = env.action_space.low.astype(np.float64).flatten()
    highs = env.action_space.high.astype(np.float64).flatten()

    # 3) 启动 GUI（滑块范围与动作空间一致）
    print(act_dim)
    tuner = ActionTunerRunner(dim=act_dim, lows=lows, highs=highs, title="Action Tuner (ManiSkill)")

    # 4) 重置环境
    obs, _ = env.reset(seed=0)
    print("Env reset. Use the sliders to control actions. Close the window to exit.")

    # 5) 控制循环：从 GUI 读 action，送入 env.step
    #    如果你的控制频率希望更低一点，可以把 sleep 调大，比如 0.02（50Hz）
    try:
        while tuner.is_alive():
            # 从 GUI 读 action；并做一次 clip（以防拖动溢出）
            action = tuner.get_action()
            action = np.clip(action, lows, highs).astype(np.float32)

            # 与 Gymnasium API 对齐
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            # done = bool(terminated or truncated)
            # if done:
            #     obs, _ = env.reset()

            # 稍微睡一下，避免 CPU 打满
            time.sleep(0.01)
    finally:
        env.close()
        print("Env closed. Bye!")

if __name__ == "__main__":
    main()

