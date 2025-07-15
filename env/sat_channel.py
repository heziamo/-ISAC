# sat_channel.py
import numpy as np
from typing import Optional, Dict, Any
from scipy.constants import speed_of_light as c

class SatelliteChannel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        self.config = {
            "comm_freq": 12e9,   # 通信频率 (Hz)
            "tx_gain": 30.0,     # 发射增益 (dB)
            "rx_gain": 25.0,     # 接收增益 (dB)
            "noise_temp": 290.0  # 噪声温度 (K)
        }
        if config:
            self.config.update(config)
        
        # 当前SNR值
        self.current_snr = 0.0
        
    def update(self, distance_km: float, Pt: float, B: float) -> float:
        """
        计算卫星链路的 SNR
        
        参数:
            distance_km : 卫星距离 (km)
            Pt : 发射功率 (W)
            B : 带宽 (Hz)
            
        返回:
            SNR (dB)
        """
        # 获取配置参数
        freq = self.config["comm_freq"]
        tx_gain = self.config["tx_gain"]
        rx_gain = self.config["rx_gain"]
        T = self.config["noise_temp"]
        
        # 转换为线性值
        wavelength = c / freq
        d = distance_km * 1e3  # 转换为米
        Gt = 10 ** (tx_gain / 10)  # dB -> 线性值
        Gr = 10 ** (rx_gain / 10)  # dB -> 线性值
        
        # 自由空间路径损耗 (FSPL)
        fspl = (4 * np.pi * d / wavelength) ** 2
        
        # 接收功率 (W)
        Pr = Pt * Gt * Gr * (wavelength ** 2) / fspl
        
        # 噪声功率 (W)
        k = 1.38e-23  # 玻尔兹曼常数
        N = k * T * B
        
        # SNR (线性值 -> dB)
        snr_linear = Pr / N
        self.current_snr = 10 * np.log10(snr_linear)
        return self.current_snr
    
    def check(self) -> Dict[str, Any]:
        """系统自检"""
        return {
            "status": "OK",
            "messages": ["SatelliteChannel operational"],
            "config": self.config
        }
    
    # 保留原始函数用于测试
    @staticmethod
    def satellite_snr_model(distance_km, Pt=10, Gt=1, Gr=1, frequency=12e9, T=290, B=10e6):
        """
        计算卫星链路的 SNR（信噪比）
        
        参数:
            distance_km : 卫星距离（km）
            Pt : 发射功率（W），默认 10W
            Gt, Gr : 发射/接收天线增益（线性值），默认 1（0dBi）
            frequency : 载波频率（Hz），默认 12GHz（Ku波段）
            T : 系统噪声温度（K），默认 290K（室温）
            B : 带宽（Hz），默认 10MHz
        
        返回:
            SNR (dB)
        """
        wavelength = c / frequency  # 波长 (m)
        d = distance_km * 1e3  # 转换为米
        
        # 自由空间路径损耗 (FSPL)
        fspl = (4 * np.pi * d / wavelength) ** 2
        
        # 接收功率 (W)
        Pr = Pt * Gt * Gr * (wavelength ** 2) / fspl
        
        # 噪声功率 (W)
        k = 1.38e-23  # 玻尔兹曼常数
        N = k * T * B
        
        # SNR (线性值 -> dB)
        snr_linear = Pr / N
        return 10 * np.log10(snr_linear)

if __name__ == "__main__":
    # 测试卫星信道模型
    print("===== 测试卫星信道模型 =====")
    sat = SatelliteChannel()
    
    # 测试不同距离的SNR
    distances = [500, 1000, 2000, 5000]
    for dist in distances:
        snr = sat.update(dist, 10, 10e6)
        print(f"距离 {dist}km 时的 SNR: {snr:.2f} dB")
    
    # 自检
    print("\n===== 自检结果 =====")
    print(sat.check())
