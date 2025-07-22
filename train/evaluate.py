# train/evaluate.py
import numpy as np
import torch
from isac_sat_env import ISAC_SatEnv
from models.ppo_policy import ActorCritic
from models.baseline_rule import RandomPolicy, GreedyPolicy, evaluate_baseline
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def load_policy(model_path, obs_dim, action_dims):
    """加载训练好的策略模型"""
    policy = ActorCritic(obs_dim, action_dims)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    return policy

def evaluate_ppo(env, policy, num_episodes=100):
    """评估PPO策略"""
    results = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        comm_success = 0
        radar_success = 0
        
        while not done:
            # 获取动作
            action, _, _ = policy.get_action(obs)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # 检查是否达到阈值
            if info['comm']['snr'] >= info['comm']['threshold']:
                comm_success += 1
            if info['radar']['snr'] >= info['radar']['threshold']:
                radar_success += 1
        
        # 计算成功率
        comm_success_rate = comm_success / steps
        radar_success_rate = radar_success / steps
        
        # 保存结果
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'avg_comm_snr': np.mean([x['comm']['snr'] for x in env.history]),
            'avg_radar_snr': np.mean([x['radar']['snr'] for x in env.history]),
            'comm_success_rate': comm_success_rate,
            'radar_success_rate': radar_success_rate
        })
        
        if (episode + 1) % 10 == 0:
            print(f"Evaluated {episode+1}/{num_episodes} episodes")
    
    return results

def compare_strategies(config=None, num_episodes=100):
    """比较所有策略的性能"""
    env = ISAC_SatEnv(config)
    
    # 确定观测和动作维度
    obs_dim = env.observation_space.shape[0]
    action_dims = [
        env.action_space.spaces[0].shape[0],  # 功率
        env.action_space.spaces[1].n,         # 子载波
        env.action_space.spaces[2].shape[0]    # 卸载比例
    ]
    
    # 评估所有策略
    strategies = {
        "Random": RandomPolicy(env.action_space),
        "Greedy": GreedyPolicy(config if config else env.config),
        "PPO": load_policy("models/ppo_final.pth", obs_dim, action_dims)
    }
    
    all_results = {}
    
    for name, policy in strategies.items():
        print(f"Evaluating {name} strategy...")
        
        if name == "PPO":
            results = evaluate_ppo(env, policy, num_episodes)
        else:
            results = evaluate_baseline(env, policy, num_episodes)
        
        all_results[name] = results
        
        # 保存结果
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df.to_csv(f"results/{name.lower()}_results.csv", index=False)
    
    # 分析结果
    analyze_results(all_results)
    
    return all_results

def analyze_results(results):
    """分析并可视化结果"""
    # 创建比较图
    plt.figure(figsize=(14, 10))
    
    # 奖励比较
    plt.subplot(2, 2, 1)
    for name, res in results.items():
        rewards = [r['total_reward'] for r in res]
        plt.plot(rewards, label=name)
    plt.title("Episode Rewards Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    
    # 平均SNR比较
    plt.subplot(2, 2, 2)
    snr_data = []
    for name, res in results.items():
        comm_snr = np.mean([r['avg_comm_snr'] for r in res])
        radar_snr = np.mean([r['avg_radar_snr'] for r in res])
        snr_data.append([name, comm_snr, radar_snr])
    
    snr_df = pd.DataFrame(snr_data, columns=['Strategy', 'Comm SNR', 'Radar SNR'])
    snr_df.plot(x='Strategy', y=['Comm SNR', 'Radar SNR'], kind='bar', ax=plt.gca())
    plt.title("Average SNR Comparison")
    plt.ylabel("SNR (dB)")
    plt.grid(True)
    
    # 成功率比较
    plt.subplot(2, 2, 3)
    success_data = []
    for name, res in results.items():
        comm_success = np.mean([r['comm_success_rate'] for r in res]) * 100
        radar_success = np.mean([r['radar_success_rate'] for r in res]) * 100
        success_data.append([name, comm_success, radar_success])
    
    success_df = pd.DataFrame(success_data, columns=['Strategy', 'Comm Success', 'Radar Success'])
    success_df.plot(x='Strategy', y=['Comm Success', 'Radar Success'], kind='bar', ax=plt.gca())
    plt.title("Success Rate Comparison")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)
    
    # 奖励分布
    plt.subplot(2, 2, 4)
    reward_data = []
    for name, res in results.items():
        rewards = [r['total_reward'] for r in res]
        reward_data.append(rewards)
    
    plt.boxplot(reward_data, labels=results.keys())
    plt.title("Reward Distribution")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    # 保存并显示
    plt.tight_layout()
    plt.savefig("results/strategy_comparison.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate strategies for Satellite ISAC')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--model', type=str, default='ppo_final.pth', help='Path to PPO model')
    args = parser.parse_args()
    
    # 加载配置（如果有）
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # 运行比较
    compare_strategies(config, args.episodes)

if __name__ == "__main__":
    main()
