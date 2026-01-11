# isac_sat_env.py
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from collections import deque
from typing import Tuple, Dict, Any, Optional
from env.sat_channel import SatelliteChannel
from env.radar_echo import RadarEcho

class ISAC_SatEnv(gym.Env):
    """集成外部模块的ISAC环境"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super(ISAC_SatEnv, self).__init__()
        
        # 合并配置
        self.config = {
            # 公共参数
            "max_steps": 1000,
            "total_power": 20.0,
            "total_bandwidth": 100e6,
            
            # 通信默认参数
            "comm_freq": 12e9,
            "tx_gain": 30.0,
            "rx_gain": 25.0,
            "comm_noise_temp": 290.0,
            "comm_snr_thresh": 10.0,
            
            # 雷达默认参数
            "radar_freq": 10e9,
            "radar_gain": 35.0,
            "target_rcs": 1.0,
            "radar_noise_temp": 290.0,
            "radar_snr_thresh": -15.0,
            
            # 目标参数
            "init_distance": 500.0,
            "target_speed": 7.8  # km/s
        }
        if config:
            self.config.update(config)

        # 初始化子模块
        self.comm = SatelliteChannel({
            "comm_freq": self.config["comm_freq"],
            "tx_gain": self.config["tx_gain"],
            "rx_gain": self.config["rx_gain"],
            "noise_temp": self.config["comm_noise_temp"]
        })
        
        self.radar = RadarEcho({
            "radar_freq": self.config["radar_freq"],
            "radar_gain": self.config["radar_gain"],
            "target_rcs": self.config["target_rcs"],
            "noise_temp": self.config["radar_noise_temp"]
        })

        # 动作空间：[comm_power_ratio, comm_bw_ratio]
        self.action_space = spaces.Box(
            low=0.1, high=0.9, shape=(2,), dtype=np.float32
        )

        # 观测空间设计
        self.observation_space = spaces.Box(
            low=np.array([-30, -30, 0, 0.1, 0.1]),
            high=np.array([50, 50, 1, 0.9, 0.9]),
            dtype=np.float32
        )

        # 环境状态
        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境状态"""
        self.current_step = 0
        self.target_distance = self.config["init_distance"]
        self.last_action = np.array([0.5, 0.5])
        self.history = deque(maxlen=1000)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        # 1. 动作处理
        action = np.clip(action, 0.1, 0.9)
        self.last_action = action.copy()
        
        # 2. 资源分配
        comm_power = self.config["total_power"] * action[0]
        radar_power = self.config["total_power"] * (1 - action[0])
        comm_bw = self.config["total_bandwidth"] * action[1]
        radar_bw = self.config["total_bandwidth"] * (1 - action[1])

        # 3. 更新目标距离
        self.target_distance += self.config["target_speed"] * 1.0  # 假设1秒步长
        self.target_distance = max(100, self.target_distance)  # 最小距离限制

        # 4. 更新子系统
        comm_snr = self.comm.update(self.target_distance, comm_power, comm_bw)
        radar_snr = self.radar.update(self.target_distance, radar_power, radar_bw)
        
        # 5. 计算奖励
        reward = self._calc_reward(comm_snr, radar_snr)
        
        # 6. 记录信息
        info = {
            "comm": {
                "snr": comm_snr,
                "power": comm_power,
                "bandwidth": comm_bw,
                "threshold": self.config["comm_snr_thresh"]
            },
            "radar": {
                "snr": radar_snr,
                "power": radar_power,
                "bandwidth": radar_bw,
                "threshold": self.config["radar_snr_thresh"]
            },
            "distance": self.target_distance
        }
        self.history.append(info)
        
        # 7. 更新状态
        self.current_step += 1
        done = self.current_step >= self.config["max_steps"]
        
        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        """构建观测向量"""
        return np.array([
            self.comm.current_snr,
            self.radar.current_snr,
            float(self.current_step) / self.config["max_steps"],
            self.last_action[0],
            self.last_action[1]
        ], dtype=np.float32)

    def _calc_reward(self, comm_snr: float, radar_snr: float) -> float:
        """综合奖励函数"""
        # 性能奖励
        comm_reward = np.clip((comm_snr - 5) / 25, 0, 1)  # 5-30dB -> 0-1
        radar_reward = np.clip((radar_snr + 20) / 40, 0, 1)  # -20-20dB -> 0-1
        
        # 阈值惩罚
        comm_penalty = -5 if comm_snr < self.config["comm_snr_thresh"] else 0
        radar_penalty = -3 if radar_snr < self.config["radar_snr_thresh"] else 0
        
        # 动作平滑奖励
        action_penalty = -0.1 * np.abs(self.last_action[0] - 0.5)  # 鼓励均衡分配
        
        return 0.5*comm_reward + 0.5*radar_reward + comm_penalty + radar_penalty + action_penalty

    def render(self, mode='human'):
        """可视化"""
        if not self.history:
            return None
            
        plt.figure(figsize=(15, 8))
        
        # SNR曲线
        plt.subplot(2, 2, 1)
        plt.plot([x["comm"]["snr"] for x in self.history], 'b-', label='Comm')
        plt.plot([x["radar"]["snr"] for x in self.history], 'r-', label='Radar')
        plt.axhline(self.config["comm_snr_thresh"], color='b', linestyle='--')
        plt.axhline(self.config["radar_snr_thresh"], color='r', linestyle='--')
        plt.title("SNR Performance")
        plt.legend()
        
        # 资源分配
        plt.subplot(2, 2, 2)
        plt.stackplot(
            range(len(self.history)),
            [x["comm"]["power"] for x in self.history],
            [x["radar"]["power"] for x in self.history],
            labels=['Comm', 'Radar']
        )
        plt.title("Power Allocation")
        
        # 带宽分配
        plt.subplot(2, 2, 3)
        plt.stackplot(
            range(len(self.history)),
            [x["comm"]["bandwidth"]/1e6 for x in self.history],
            [x["radar"]["bandwidth"]/1e6 for x in self.history],
            labels=['Comm', 'Radar']
        )
        plt.title("Bandwidth Allocation (MHz)")
        
        # 目标距离
        plt.subplot(2, 2, 4)
        plt.plot([x["distance"] for x in self.history], 'k-')
        plt.title("Target Distance (km)")
        
        plt.tight_layout()
        if mode == 'human':
            plt.show()
        else:
            plt.close()
            return plt.gcf()

    def check(self) -> Tuple[bool, dict]:
        """系统自检"""
        # 检查子模块
        comm_check = self.comm.check()
        radar_check = self.radar.check()
        
        # 检查动作空间
        action_valid = True  # Simplified validation
        
        # 综合报告
        report = {
            "modules": {
                "sat_channel": comm_check,
                "radar_echo": radar_check
            },
            "action_space": {
                "valid": action_valid,
                "low": [0.1, 0.1],
                "high": [0.9, 0.9]
            },
            "config": self.config
        }
        
        is_ok = all([
            comm_check["status"] in ["OK", "WARNING"],
            radar_check["status"] == "OK",
            action_valid
        ])
        
        return is_ok, report

# 环境注册函数
def make_env(config=None):
    def _init():
        env = ISAC_SatEnv(config)
        return env
    return _init
