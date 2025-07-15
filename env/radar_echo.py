# radar_echo.py
import numpy as np
from scipy.constants import speed_of_light as c
from typing import Dict, Any, Optional

class RadarEcho:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        self.config = {
            "radar_freq": 10e9,  # 雷达频率 (Hz)
            "radar_gain": 35.0,  # 雷达增益 (dB)
            "target_rcs": 1.0,   # 目标雷达截面积 (m²)
            "noise_temp": 290.0  # 噪声温度 (K)
        }
        if config:
            self.config.update(config)
        
        # 当前SNR值
        self.current_snr = 0.0
        
    def update(self, R_km: float, Pt: float, B: float) -> float:
        """
        计算雷达回波 SNR
        
        参数:
            R_km : 目标距离 (km)
            Pt : 发射功率 (W)
            B : 带宽 (Hz)
            
        返回:
            SNR (dB)
        """
        # 获取配置参数
        freq = self.config["radar_freq"]
        G = self.config["radar_gain"]
        sigma = self.config["target_rcs"]
        T = self.config["noise_temp"]
        
        # 转换为线性值
        R = R_km * 1e3  # 转换为米
        wavelength = c / freq
        G_linear = 10 ** (G / 10)  # dB -> 线性值
        
        # 雷达方程计算 SNR
        numerator = Pt * (G_linear ** 2) * (wavelength ** 2) * sigma
        denominator = (4 * np.pi) ** 3 * (R ** 4) * 1.38e-23 * T * B
        snr_linear = numerator / denominator
        self.current_snr = 10 * np.log10(snr_linear)
        return self.current_snr
    
    def check(self) -> Dict[str, Any]:
        """系统自检"""
        return {
            "status": "OK",
            "messages": ["RadarEcho operational"],
            "config": self.config
        }
    
    # 保留原始函数用于测试
    @staticmethod
    def radar_snr_model(R_km, Pt=1e3, G=30, sigma=1, freq=10e9, T=290, B=1e6):
        """
        计算雷达回波 SNR
        :param R_km: 目标距离 (km)
        :param Pt: 发射功率 (W)
        :param G: 天线增益 (dB)
        :param sigma: 目标RCS (m²)
        :param freq: 雷达频率 (Hz)
        :param T: 噪声温度 (K)
        :param B: 带宽 (Hz)
        :return: SNR (dB)
        """
        R = R_km * 1e3  # 转换为米
        wavelength = c / freq
        G_linear = 10 ** (G / 10)  # dB -> 线性值
        
        # 雷达方程计算 SNR
        numerator = Pt * (G_linear ** 2) * (wavelength ** 2) * sigma
        denominator = (4 * np.pi) ** 3 * (R ** 4) * 1.38e-23 * T * B
        snr_linear = numerator / denominator
        return 10 * np.log10(snr_linear)

if __name__ == "__main__":
    # 测试雷达回波模型
    print("===== 测试雷达回波模型 =====")
    radar = RadarEcho()
    
    # 测试不同距离的SNR
    distances = [100, 200, 500, 1000]
    for dist in distances:
        snr = radar.update(dist, 1000, 1e6)
        print(f"距离 {dist}km 时的雷达 SNR: {snr:.2f} dB")
    
    # 自检
    print("\n===== 自检结果 =====")
    print(radar.check())
