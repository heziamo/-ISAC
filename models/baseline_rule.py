# baseline_rule.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import csv
import os
from isac_sat_env import ISAC_SatEnv
from typing import Dict, List, Tuple, Any

def random_policy(env) -> np.ndarray:
    """随机策略：在动作空间内均匀随机选择动作"""
    return env.action_space.sample()

def greedy_policy(env) -> np.ndarray:
    """
    贪心策略：
    - 通信功率分配：优先满足通信需求（50%功率）
    - 带宽分配：优先满足雷达需求（50%带宽）
    """
    # 使用固定比例的策略
    return np.array([0.5, 0.5], dtype=np.float32)

def run_episode(env, policy_fn) -> Dict[str, Any]:
    """运行单个episode并收集性能指标"""
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    comm_snr_list = []
    radar_snr_list = []
    comm_success_count = 0
    radar_success_count = 0
    
    while not done:
        # 选择动作
        action = policy_fn(env)
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 更新统计信息
        total_reward += reward
        steps += 1
        comm_snr_list.append(info['comm']['snr'])
        radar_snr_list.append(info['radar']['snr'])
        
        # 检查是否达到阈值
        if info['comm']['snr'] >= info['comm']['threshold']:
            comm_success_count += 1
        if info['radar']['snr'] >= info['radar']['threshold']:
            radar_success_count += 1
    
    # 计算平均性能指标
    avg_comm_snr = np.mean(comm_snr_list) if comm_snr_list else 0
    avg_radar_snr = np.mean(radar_snr_list) if radar_snr_list else 0
    comm_success_rate = comm_success_count / steps if steps > 0 else 0
    radar_success_rate = radar_success_count / steps if steps > 0 else 0
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'avg_comm_snr': avg_comm_snr,
        'avg_radar_snr': avg_radar_snr,
        'comm_success_rate': comm_success_rate,
        'radar_success_rate': radar_success_rate
    }

def run_baseline(policy_name: str, num_episodes: int) -> List[Dict[str, Any]]:
    """运行指定策略的多个episode"""
    env = ISAC_SatEnv()
    results = []
    
    # 选择策略
    if policy_name == 'random':
        policy_fn = random_policy
    elif policy_name == 'greedy':
        policy_fn = greedy_policy
    else:
        raise ValueError(f"未知策略: {policy_name}")
    
    print(f"开始运行 {policy_name} 策略 ({num_episodes} episodes)...")
    
    for episode in range(num_episodes):
        metrics = run_episode(env, policy_fn)
        metrics['episode'] = episode
        results.append(metrics)
        
        # 每10个episode打印进度
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes} 完成. "
                  f"平均奖励: {np.mean([r['total_reward'] for r in results[-10:]]):.2f}")
    
    return results

def save_results_to_csv(results: List[Dict[str, Any]], filename: str):
    """将结果保存到CSV文件"""
    # 确保结果目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'total_reward', 'steps', 
                      'avg_comm_snr', 'avg_radar_snr',
                      'comm_success_rate', 'radar_success_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"结果已保存至: {filename}")

def main():
    """主函数：运行两种基线策略并保存结果"""
    # 运行随机策略
    random_results = run_baseline('random', 100)
    save_results_to_csv(random_results, 'results/random_baseline.csv')
    
    # 运行贪心策略
    greedy_results = run_baseline('greedy', 100)
    save_results_to_csv(greedy_results, 'results/greedy_baseline.csv')
    
    # 打印摘要
    print("\n性能摘要:")
    print(f"随机策略 - 平均奖励: {np.mean([r['total_reward'] for r in random_results]):.2f}")
    print(f"贪心策略 - 平均奖励: {np.mean([r['total_reward'] for r in greedy_results]):.2f}")

if __name__ == "__main__":
    main()