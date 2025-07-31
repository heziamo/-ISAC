# test_integrated_visualization.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import wandb
from env.radar_echo import RadarEcho
from env.sat_channel import SatelliteChannel
from env.isac_sat_env import ISAC_SatEnv

def test_radar_visualization():
    """测试雷达模块可视化"""
    print("=== 测试雷达模块可视化 ===")
    
    # 初始化WandB
    wandb.init(project="satellite-isac", 
              name="radar_visualization_test",
              config={
                  "radar_freq": 10e9,
                  "radar_gain": 35.0,
                  "target_rcs": 1.0,
                  "noise_temp": 290.0,
                  "radar_snr_thresh": -15.0
              })
    
    # 创建雷达模型
    radar = RadarEcho(wandb.config.as_dict())
    
    # 测试SNR曲线
    distances = np.linspace(100, 1000, 20)
    snr_values = radar.plot_snr_vs_distance(
        distances, 
        Pt=1000, 
        B=1e6,
        title="雷达SNR性能曲线 (L波段)"
    )
    print(f"SNR测试完成，最大SNR: {max(snr_values):.2f} dB")
    
    # 测试距离FFT图
    test_distances = [200, 500, 800]
    for dist in test_distances:
        fft_result = radar.plot_range_profile(
            dist, 
            Pt=1000, 
            B=1e6,
            num_samples=2048
        )
        print(f"距离 {dist}km 的FFT测试完成，峰值位置: {np.argmax(fft_result)}")
    
    wandb.finish()

def test_channel_visualization():
    """测试卫星信道可视化"""
    print("\n=== 测试卫星信道可视化 ===")
    
    # 初始化WandB
    wandb.init(project="satellite-isac", 
              name="channel_visualization_test",
              config={
                  "frequency": 12e9,
                  "tx_gain": 30.0,
                  "rx_gain": 25.0,
                  "noise_temp": 290.0,
                  "comm_snr_thresh": 10.0,
                  "rain_attenuation": 0.15,
                  "atmospheric_loss": 0.03
              })
    
    # 创建信道模型
    channel = SatelliteChannel(wandb.config.as_dict())
    
    # 测试SNR曲线
    distances = np.linspace(100, 2000, 20)
    snr_values = channel.plot_snr_vs_distance(
        distances, 
        Pt=100, 
        B=10e6,
        title="卫星通信SNR性能曲线 (Ku波段)"
    )
    print(f"通信SNR测试完成，最大SNR: {max(snr_values):.2f} dB")
    
    # 测试链路预算分析
    for dist in [500, 1000, 2000]:
        _ = channel.plot_link_budget(
            dist, 
            Pt=100, 
            B=10e6
        )
        print(f"距离 {dist}km 的链路预算分析完成")
    
    wandb.finish()

def create_heatmap(data, row_labels, col_labels, title):
    """创建热力图并返回Matplotlib图像"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="YlGn")
    
    # 显示数值
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                         ha="center", va="center", color="black")
    
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def test_env_visualization():
    """测试ISAC环境可视化"""
    print("\n=== 测试ISAC环境可视化 ===")
    
    # 初始化WandB
    wandb.init(project="satellite-isac", 
              name="isac_env_test",
              config={
                  "num_targets": 2,
                  "max_steps": 200,
                  "total_power": 20.0,
                  "total_bandwidth": 100e6,
                  "comm_freq": 12e9,
                  "radar_freq": 10e9,
                  "target_speed": 7.8
              })
    
    # 创建环境
    env = ISAC_SatEnv(wandb.config.as_dict())
    env.enable_wandb_logging()
    
    print("环境配置:", wandb.config)
    
    # 忽略Gym的float32警告
    import warnings
    warnings.filterwarnings("ignore", message="Box bound precision lowered by casting to float32")
    
    # 随机策略测试
    obs = env.reset()
    for i in range(wandb.config.max_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # 每50步记录一次详细数据
        if i % 50 == 0:
            # 创建自定义热力图
            power_allocation = np.array([[t["comm"]["power"], t["radar"]["power"]] for t in info["targets"]])
            bw_allocation = np.array([[t["comm"]["bandwidth"], t["radar"]["bandwidth"]] for t in info["targets"]])
            
            # 功率分配热力图
            power_fig = create_heatmap(
                power_allocation,
                ["目标1", "目标2"],
                ["通信", "雷达"],
                "功率分配 (W)"
            )
            
            # 带宽分配热力图
            bw_fig = create_heatmap(
                bw_allocation / 1e6,  # 转换为MHz
                ["目标1", "目标2"],
                ["通信", "雷达"],
                "带宽分配 (MHz)"
            )
            
            wandb.log({
                "power_allocation": wandb.Image(power_fig),
                "bandwidth_allocation": wandb.Image(bw_fig),
                "step": i
            })
            
            plt.close('all')
            
            # 渲染环境状态
            env.render(mode='rgb_array')
        
        if done:
            obs = env.reset()
    
    print("环境测试完成，最终奖励:", reward)
    
    # 保存最终状态图
    final_fig = env.render(mode='rgb_array')
    if final_fig:
        wandb.log({"final_state": wandb.Image(final_fig)})
    
    wandb.finish()

def test_comparative_analysis():
    """对比分析雷达与通信性能"""
    print("\n=== 对比分析雷达与通信性能 ===")
    
    # 初始化WandB
    wandb.init(project="satellite-isac", 
              name="comparative_analysis",
              config={
                  "distance_range": [100, 1000],
                  "test_points": 20,
                  "radar_power": 500,
                  "comm_power": 100,
                  "bandwidth": 10e6
              })
    
    # 创建模型实例
    radar = RadarEcho({
        "radar_freq": 10e9,
        "radar_gain": 35.0,
        "target_rcs": 1.0,
        "noise_temp": 290.0
    })
    
    channel = SatelliteChannel({
        "frequency": 12e9,
        "tx_gain": 30.0,
        "rx_gain": 25.0,
        "noise_temp": 290.0
    })
    
    # 生成测试数据
    distances = np.linspace(
        wandb.config.distance_range[0],
        wandb.config.distance_range[1],
        wandb.config.test_points
    )
    
    radar_snrs = []
    comm_snrs = []
    
    for dist in distances:
        radar_snrs.append(radar.update(dist, wandb.config.radar_power, wandb.config.bandwidth))
        comm_snrs.append(channel.update(dist, wandb.config.comm_power, wandb.config.bandwidth))
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.plot(distances, radar_snrs, 'b-', label='雷达SNR')
    plt.plot(distances, comm_snrs, 'g-', label='通信SNR')
    plt.axhline(y=-15, color='b', linestyle='--', label='雷达SNR阈值')
    plt.axhline(y=10, color='g', linestyle='--', label='通信SNR阈值')
    plt.xlabel('距离 (km)')
    plt.ylabel('SNR (dB)')
    plt.title('雷达与通信SNR性能对比')
    plt.legend()
    plt.grid(True)
    
    # 记录到WandB
    wandb.log({
        "radar_vs_comm_snr": wandb.Image(plt),
        "radar_snr_at_500km": radar_snrs[np.argmin(np.abs(distances-500))],
        "comm_snr_at_500km": comm_snrs[np.argmin(np.abs(distances-500))]
    })
    
    plt.close()
    wandb.finish()

if __name__ == "__main__":
    # 执行所有测试
    test_radar_visualization()
    test_channel_visualization()
    test_env_visualization()
    test_comparative_analysis()
    
    print("\n===== 所有测试完成 =====")