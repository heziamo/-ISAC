# radar_echo.py
import numpy as np
from scipy.constants import speed_of_light as c
from scipy.fft import fft
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any, Optional

class RadarEcho:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        self.config = {
            "radar_freq": 10e9,  # 雷达频率 (Hz)
            "radar_gain": 35.0,  # 雷达增益 (dB)
            "target_rcs": 1.0,   # 目标雷达截面积 (m²)
            "noise_temp": 290.0,  # 噪声温度 (K)
            "radar_snr_thresh": -15.0  # SNR阈值 (dB)
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
    
    def plot_snr_vs_distance(self, distances_km: list, Pt: float, B: float, 
                           save_to_wandb: bool = True, title: str = None):
        """
        绘制SNR随距离变化曲线
        """
        snrs = [self.update(d, Pt, B) for d in distances_km]
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances_km, snrs, 'b-o', linewidth=2)
        plt.axhline(y=self.config["radar_snr_thresh"], color='r', linestyle='--', 
                   label=f'SNR阈值({self.config["radar_snr_thresh"]} dB)')
        plt.xlabel('距离 (km)')
        plt.ylabel('SNR (dB)')
        plt.title(title or f'雷达SNR vs 距离 (Pt={Pt}W, B={B/1e6}MHz)')
        plt.grid(True)
        plt.legend()
        
        if save_to_wandb and wandb.run:
            wandb.log({"radar_snr_vs_distance": wandb.Image(plt)})
        
        plt.close()
        return snrs
    
    def plot_range_profile(self, R_km: float, Pt: float, B: float, 
                         num_samples: int = 1024, save_to_wandb: bool = True):
        """
        生成距离FFT图(模拟雷达距离像)
        """
        # 模拟雷达回波信号
        fs = 2 * B  # 采样频率
        t = np.linspace(0, 1/B, num_samples)
        
        # 生成LFM信号 (简化模型)
        target_R = R_km * 1e3
        delay = 2 * target_R / c  # 双程延迟
        chirp_rate = B / (1/B)  # 调频斜率
        
        # 模拟回波信号 (加入噪声)
        signal = np.exp(1j * 2 * np.pi * (self.config["radar_freq"] * t + 0.5 * chirp_rate * (t-delay)**2))
        noise_power = 10 ** (-self.update(R_km, Pt, B) / 10)  # 根据SNR计算噪声功率
        signal += np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        # 计算FFT
        fft_result = np.abs(fft(signal))
        range_bins = np.linspace(0, c/(2*B), num_samples) / 1000  # 转换为km
        
        # 绘制FFT结果
        plt.figure(figsize=(10, 6))
        plt.plot(range_bins, 20 * np.log10(fft_result), linewidth=1.5)
        plt.axvline(x=R_km, color='r', linestyle='--', label=f'真实距离: {R_km}km')
        plt.xlabel('距离 (km)')
        plt.ylabel('幅度 (dB)')
        plt.title(f'雷达距离像 (目标距离: {R_km}km, 分辨率: {c/(2*B)/1000:.2f}km)')
        plt.grid(True)
        plt.legend()
        
        if save_to_wandb and wandb.run:
            wandb.log({
                "radar_range_profile": wandb.Image(plt),
                "target_distance": R_km,
                "range_resolution": c/(2*B)/1000
            })
        
        plt.close()
        return fft_result
    
    def check(self) -> Dict[str, Any]:
        """系统自检"""
        return {
            "status": "OK",
            "messages": ["RadarEcho operational"],
            "config": self.config
        }
    
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
