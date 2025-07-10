import numpy as np
import matplotlib.pyplot as plt
import wandb

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
    c = 3e8  # 光速 (m/s)
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
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db

def plot_snr_vs_distance():
    """
    生成 SNR 随距离变化的曲线，并记录到 wandb
    """
    # 初始化 wandb 实验
    wandb.init(
        project="satellite-channel",
        name="snr_vs_distance",
        config={
            "Pt": 10,
            "frequency_GHz": 12,
            "bandwidth_MHz": 10,
            "noise_temp_K": 290
        }
    )
    
    # 模拟不同距离下的 SNR（100km ~ 40000km）
    distances = np.linspace(100, 40000, 100)
    snr_values = [satellite_snr_model(d) for d in distances]
    
    # 绘制 SNR 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(distances, snr_values, 'b-', linewidth=2, label="SNR")
    plt.xlabel("Satellite Distance (km)")
    plt.ylabel("SNR (dB)")
    plt.title("Satellite Link SNR vs. Distance")
    plt.grid(True)
    plt.legend()
    
    # 记录到 wandb
    wandb.log({
        "SNR_curve": wandb.Image(plt),
        "max_SNR": max(snr_values),
        "min_SNR": min(snr_values)
    })
    
    # 显示图表并关闭 wandb
    plt.show()
    wandb.finish()

if __name__ == "__main__":
    # 示例：运行 SNR 曲线分析
    plot_snr_vs_distance()
