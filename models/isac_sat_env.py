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
            "target_speed": 7.8,  # km/s
            
            # 新增：目标数量（卫星数量）
            "num_targets": 2
        }
        if config:
            self.config.update(config)

        self.num_targets = self.config.get("num_targets", 2)
        # 初始化多个目标
        self.comm = [SatelliteChannel({
            "comm_freq": self.config["comm_freq"],
            "tx_gain": self.config["tx_gain"],
            "rx_gain": self.config["rx_gain"],
            "noise_temp": self.config["comm_noise_temp"]
        }) for _ in range(self.num_targets)]
        self.radar = [RadarEcho({
            "radar_freq": self.config["radar_freq"],
            "radar_gain": self.config["radar_gain"],
            "target_rcs": self.config["target_rcs"],
            "noise_temp": self.config["radar_noise_temp"]
        }) for _ in range(self.num_targets)]

        # 动作空间扩展：每个目标分配功率和带宽
        self.action_space = spaces.Box(
            low=0.1, high=0.9, shape=(self.num_targets, 2), dtype=np.float32
        )

        # 观测空间扩展
        self.observation_space = spaces.Box(
            low=np.array([-30] * self.num_targets + [-30] * self.num_targets + [0] + [0.1] * self.num_targets * 2),
            high=np.array([50] * self.num_targets + [50] * self.num_targets + [1] + [0.9] * self.num_targets * 2),
            dtype=np.float32
        )

        # 环境状态
        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境状态"""
        self.current_step = 0
        self.target_distance = [self.config["init_distance"]] * self.num_targets
        self.last_action = np.array([[0.5, 0.5]] * self.num_targets)
        self.history = deque(maxlen=1000)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        # 1. 动作处理
        action = np.clip(action, 0.1, 0.9)
        self.last_action = action.copy()
        comm_powers = []
        radar_powers = []
        comm_bws = []
        radar_bws = []
        comm_snrs = []
        radar_snrs = []
        info_targets = []

        total_power = self.config["total_power"]
        total_bw = self.config["total_bandwidth"]

        # 均分资源（或可自定义分配策略）
        for i in range(self.num_targets):
            comm_power = total_power * action[i, 0] / self.num_targets
            radar_power = total_power * (1 - action[i, 0]) / self.num_targets
            comm_bw = total_bw * action[i, 1] / self.num_targets
            radar_bw = total_bw * (1 - action[i, 1]) / self.num_targets

            self.target_distance[i] += self.config["target_speed"] * 1.0
            self.target_distance[i] = max(100, self.target_distance[i])

            comm_snr = self.comm[i].update(self.target_distance[i], comm_power, comm_bw)
            radar_snr = self.radar[i].update(self.target_distance[i], radar_power, radar_bw)

            comm_powers.append(comm_power)
            radar_powers.append(radar_power)
            comm_bws.append(comm_bw)
            radar_bws.append(radar_bw)
            comm_snrs.append(comm_snr)
            radar_snrs.append(radar_snr)

            info_targets.append({
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
                "distance": self.target_distance[i]
            })

        # 公平性指标（Jain's index）
        fairness_comm = self._jain_index(comm_snrs)
        fairness_radar = self._jain_index(radar_snrs)

        # 奖励：所有目标平均 + 公平性奖励
        reward = self._calc_reward(comm_snrs, radar_snrs) + 0.2 * (fairness_comm + fairness_radar)

        info = {
            "targets": info_targets,
            "fairness_comm": fairness_comm,
            "fairness_radar": fairness_radar
        }
        self.history.append(info)
        self.current_step += 1
        done = self.current_step >= self.config["max_steps"]
        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        obs = []
        obs += [c.current_snr for c in self.comm]
        obs += [r.current_snr for r in self.radar]
        obs.append(float(self.current_step) / self.config["max_steps"])
        obs += self.last_action.flatten().tolist()
        return np.array(obs, dtype=np.float32)

    def _calc_reward(self, comm_snrs, radar_snrs) -> float:
        comm_rewards = [np.clip((snr - 5) / 25, 0, 1) for snr in comm_snrs]
        radar_rewards = [np.clip((snr + 20) / 40, 0, 1) for snr in radar_snrs]
        comm_penalty = sum([-5 if snr < self.config["comm_snr_thresh"] else 0 for snr in comm_snrs])
        radar_penalty = sum([-3 if snr < self.config["radar_snr_thresh"] else 0 for snr in radar_snrs])
        action_penalty = -0.1 * np.sum(np.abs(self.last_action - 0.5))
        return 0.5 * np.mean(comm_rewards) + 0.5 * np.mean(radar_rewards) + comm_penalty + radar_penalty + action_penalty

    def _jain_index(self, values):
        values = np.array(values)
        if np.sum(values) == 0:
            return 0.0
        return (np.sum(values) ** 2) / (len(values) * np.sum(values ** 2) + 1e-8)

    def render(self, mode='human'):
        """可视化"""
        if not self.history:
            return None

        # 处理多目标数据
        num_targets = len(self.history[0]["targets"])
        comm_snrs = [[] for _ in range(num_targets)]
        radar_snrs = [[] for _ in range(num_targets)]
        comm_powers = [[] for _ in range(num_targets)]
        radar_powers = [[] for _ in range(num_targets)]
        comm_bws = [[] for _ in range(num_targets)]
        radar_bws = [[] for _ in range(num_targets)]
        distances = [[] for _ in range(num_targets)]

        for entry in self.history:
            for i, tgt in enumerate(entry["targets"]):
                comm_snrs[i].append(tgt["comm"]["snr"])
                radar_snrs[i].append(tgt["radar"]["snr"])
                comm_powers[i].append(tgt["comm"]["power"])
                radar_powers[i].append(tgt["radar"]["power"])
                comm_bws[i].append(tgt["comm"]["bandwidth"]/1e6)
                radar_bws[i].append(tgt["radar"]["bandwidth"]/1e6)
                distances[i].append(tgt["distance"])

        plt.figure(figsize=(15, 8))

        # SNR曲线
        plt.subplot(2, 2, 1)
        for i in range(num_targets):
            plt.plot(comm_snrs[i], label=f'Comm-{i+1}')
            plt.plot(radar_snrs[i], label=f'Radar-{i+1}')
        plt.axhline(self.config["comm_snr_thresh"], color='b', linestyle='--')
        plt.axhline(self.config["radar_snr_thresh"], color='r', linestyle='--')
        plt.title("SNR Performance")
        plt.legend()

        # 资源分配
        plt.subplot(2, 2, 2)
        for i in range(num_targets):
            plt.plot(comm_powers[i], label=f'Comm-{i+1}')
            plt.plot(radar_powers[i], label=f'Radar-{i+1}')
        plt.title("Power Allocation")
        plt.legend()

        # 带宽分配
        plt.subplot(2, 2, 3)
        for i in range(num_targets):
            plt.plot(comm_bws[i], label=f'Comm-{i+1}')
            plt.plot(radar_bws[i], label=f'Radar-{i+1}')
        plt.title("Bandwidth Allocation (MHz)")
        plt.legend()

        # 目标距离
        plt.subplot(2, 2, 4)
        for i in range(num_targets):
            plt.plot(distances[i], label=f'Target-{i+1}')
        plt.title("Target Distance (km)")
        plt.legend()

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
