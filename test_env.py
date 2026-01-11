# test_env.py
from isac_sat_env import ISAC_SatEnv
import numpy as np

def main():
    # 创建环境实例
    env = ISAC_SatEnv()
    
    # 运行环境自检
    print("===== 运行环境自检 =====")
    is_ok, report = env.check()
    print(f"环境自检结果: {'通过' if is_ok else '未通过'}")
    print("自检报告:")
    print(f"通信模块状态: {report['modules']['sat_channel']['status']}")
    print(f"雷达模块状态: {report['modules']['radar_echo']['status']}")
    print(f"动作空间有效性: {'有效' if report['action_space']['valid'] else '无效'}")
    
    # 重置环境
    print("\n===== 重置环境 =====")
    obs = env.reset()
    print(f"初始观测值: {obs}")
    
    # 运行环境
    print("\n===== 开始模拟 =====")
    done = False
    step_count = 0
    max_steps = 10  # 仅运行10步用于演示
    
    while not done and step_count < max_steps:
        # 随机动作（实际应用中应由智能体生成）
        action = env.action_space.sample()
        
        # 执行一步
        obs, reward, done, info = env.step(action)
        
        print(f"\n步骤 {step_count + 1}:")
        print(f"  动作: [功率分配比例: {action[0]:.2f}, 带宽分配比例: {action[1]:.2f}]")
        print(f"  观测值: [通信SNR: {obs[0]:.2f}dB, 雷达SNR: {obs[1]:.2f}dB, 进度: {obs[2]:.2f}, 上次功率分配: {obs[3]:.2f}, 上次带宽分配: {obs[4]:.2f}]")
        print(f"  奖励: {reward:.2f}")
        print(f"  通信状态: SNR={info['comm']['snr']:.2f}dB (阈值={info['comm']['threshold']}dB), 功率={info['comm']['power']:.2f}W, 带宽={info['comm']['bandwidth']/1e6:.2f}MHz")
        print(f"  雷达状态: SNR={info['radar']['snr']:.2f}dB (阈值={info['radar']['threshold']}dB), 功率={info['radar']['power']:.2f}W, 带宽={info['radar']['bandwidth']/1e6:.2f}MHz")
        print(f"  目标距离: {info['distance']:.2f}km")
        
        step_count += 1
    
    # 渲染结果
    print("\n===== 渲染结果 =====")
    env.render(mode='human')

if __name__ == "__main__":
    main()