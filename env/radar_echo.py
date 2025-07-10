import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.constants import speed_of_light as c

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
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db

def simulate_radar_echo():
    """ 模拟雷达回波并可视化 """
    wandb.init(project="radar-model", name="echo_snr_curve")
    
    # 模拟不同距离的目标 (1km ~ 100km)
    ranges = np.linspace(1, 100, 100)
    snr_values = [radar_snr_model(R) for R in ranges]
    
    # 添加随机噪声模拟实际回波
    noisy_snr = snr_values + np.random.normal(0, 2, len(ranges))
    
    # 绘制 SNR 曲线
    plt.figure(figsize=(12, 6))
    plt.plot(ranges, snr_values, 'b-', label="Theoretical SNR")
    plt.plot(ranges, noisy_snr, 'r--', label="Noisy SNR (Simulated)")
    plt.xlabel("Target Distance (km)")
    plt.ylabel("SNR (dB)")
    plt.title("Radar Echo SNR vs. Distance")
    plt.grid(True)
    plt.legend()
    
    # 记录到 wandb
    wandb.log({
        "radar_snr_curve": wandb.Image(plt),
        "parameters": {
            "Pt_kW": 1,
            "frequency_GHz": 10,
            "bandwidth_MHz": 1,
            "RCS_m2": 1
        }
    })
    
    plt.show()
    wandb.finish()

if __name__ == "__main__":
    simulate_radar_echo()
