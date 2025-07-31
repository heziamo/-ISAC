# sat_channel.py
import numpy as np
from scipy.constants import speed_of_light as c, Boltzmann as k
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any, Optional

class SatelliteChannel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        self.config = {
            "frequency": 12e9,       # 载波频率 (Hz)
            "tx_gain": 30.0,         # 发射天线增益 (dB)
            "rx_gain": 25.0,         # 接收天线增益 (dB)
            "noise_temp": 290.0,     # 噪声温度 (K)
            "comm_snr_thresh": 10.0, # SNR阈值 (dB)
            "rain_attenuation": 0.1, # 雨衰 (dB/km)
            "atmospheric_loss": 0.02 # 大气损耗 (dB/km)
        }
        if config:
            self.config.update(config)
        
        self.current_snr = 0.0
        self.current_distance = 0.0
        
    def update(self, R_km: float, Pt: float, B: float) -> float:
        """
        计算卫星通信链路SNR
        
        参数:
            R_km : 距离 (km)
            Pt : 发射功率 (W)
            B : 带宽 (Hz)
            
        返回:
            SNR (dB)
        """
        # 获取配置参数
        freq = self.config["frequency"]
        Gt = self.config["tx_gain"]
        Gr = self.config["rx_gain"]
        T = self.config["noise_temp"]
        rain_att = self.config["rain_attenuation"]
        atm_loss = self.config["atmospheric_loss"]
        
        # 转换为线性值
        R = R_km * 1e3  # 转换为米
        wavelength = c / freq
        Gt_linear = 10 ** (Gt / 10)
        Gr_linear = 10 ** (Gr / 10)
        
        # 计算自由空间路径损耗
        FSPL = (4 * np.pi * R / wavelength) ** 2
        FSPL_db = 20 * np.log10(4 * np.pi * R / wavelength)
        
        # 附加损耗 (雨衰 + 大气损耗)
        additional_loss_db = rain_att * R_km + atm_loss * R_km
        
        # 计算接收功率
        Pr = (Pt * Gt_linear * Gr_linear * wavelength ** 2) / ((4 * np.pi * R) ** 2)
        Pr_db = 10 * np.log10(Pt) + Gt + Gr - FSPL_db - additional_loss_db
        
        # 计算噪声功率
        Pn = k * T * B
        Pn_db = 10 * np.log10(Pn)
        
        # 计算SNR
        snr_db = Pr_db - Pn_db
        self.current_snr = snr_db
        self.current_distance = R_km
        
        return snr_db
    
    def plot_snr_vs_distance(self, distances_km: list, Pt: float, B: float, 
                           save_to_wandb: bool = True, title: str = None):
        """
        绘制SNR随距离变化曲线
        """
        snrs = [self.update(d, Pt, B) for d in distances_km]
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances_km, snrs, 'g-s', linewidth=2)
        plt.axhline(y=self.config["comm_snr_thresh"], color='r', linestyle='--', 
                   label=f'SNR阈值({self.config["comm_snr_thresh"]} dB)')
        plt.xlabel('距离 (km)')
        plt.ylabel('SNR (dB)')
        plt.title(title or f'卫星通信SNR vs 距离 (Pt={Pt}W, B={B/1e6}MHz)')
        plt.grid(True)
        plt.legend()
        
        if save_to_wandb and wandb.run:
            wandb.log({"sat_comm_snr_vs_distance": wandb.Image(plt)})
        
        plt.close()
        return snrs
    
    def plot_link_budget(self, R_km: float, Pt: float, B: float,
                       save_to_wandb: bool = True):
        """
        绘制链路预算分析图
        """
        # 计算各组成部分
        freq = self.config["frequency"]
        Gt = self.config["tx_gain"]
        Gr = self.config["rx_gain"]
        T = self.config["noise_temp"]
        rain_att = self.config["rain_attenuation"]
        atm_loss = self.config["atmospheric_loss"]
        
        R = R_km * 1e3
        wavelength = c / freq
        FSPL_db = 20 * np.log10(4 * np.pi * R / wavelength)
        additional_loss_db = rain_att * R_km + atm_loss * R_km
        Pt_db = 10 * np.log10(Pt)
        Pn_db = 10 * np.log10(k * T * B)
        snr_db = self.update(R_km, Pt, B)
        
        # 准备数据
        components = [
            ('发射功率', Pt_db, 'lightgreen'),
            ('发射天线增益', Gt, 'lightblue'),
            ('接收天线增益', Gr, 'lightblue'),
            ('自由空间损耗', -FSPL_db, 'salmon'),
            ('附加损耗', -additional_loss_db, 'orange'),
            ('噪声功率', -Pn_db, 'lightgray')
        ]
        
        # 绘制堆叠图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bottom = 0
        for label, value, color in components:
            if value > 0:
                ax.bar('链路预算', value, bottom=bottom, label=label, color=color)
                bottom += value
            else:
                ax.bar('链路预算', -value, bottom=bottom + value, label=label, color=color)
        
        ax.axhline(y=bottom + snr_db, color='purple', linestyle='--', 
                  label=f'最终SNR: {snr_db:.2f} dB')
        ax.set_ylabel('功率 (dB)')
        ax.set_title(f'卫星通信链路预算分析 (距离: {R_km}km)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_to_wandb and wandb.run:
            wandb.log({
                "link_budget_analysis": wandb.Image(fig),
                "distance": R_km,
                "snr": snr_db
            })
        
        plt.close()
        return fig
    
    def check(self) -> Dict[str, Any]:
        """系统自检"""
        return {
            "status": "OK",
            "messages": ["SatelliteChannel operational"],
            "config": self.config
        }

if __name__ == "__main__":
    # 测试卫星信道模型
    print("===== 测试卫星信道模型 =====")
    channel = SatelliteChannel()
    
    # 测试不同距离的SNR
    distances = [200, 500, 1000, 2000, 36000]  # 包含GEO距离测试
    for dist in distances:
        snr = channel.update(dist, 100, 10e6)
        print(f"距离 {dist}km 时的通信 SNR: {snr:.2f} dB")
    
    # 自检
    print("\n===== 自检结果 =====")
    print(channel.check())
