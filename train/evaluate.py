# evaluate_performance.py (最终修复版)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from baseline_rule import run_baseline
from env.isac_sat_env import ISAC_SatEnv
import torch
from models.ppo_policy import ActorCritic
import os
import logging
from typing import Dict, List, Callable, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PPOPolicyWrapper:
    """PPO策略包装器，解决current_obs问题"""
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.last_obs = None
    
    def __call__(self, env):
        with torch.no_grad():
            if self.last_obs is None:
                self.last_obs = env.reset()
            obs_tensor = torch.as_tensor(self.last_obs, dtype=torch.float32).unsqueeze(0)
            action_mu, _ = self.policy(obs_tensor)
            action = action_mu.squeeze(0).numpy()
            return action.reshape((self.env.config.get("num_targets", 2), 2))
    
    def update_obs(self, obs):
        self.last_obs = obs

def load_ppo_policy(env, model_path: str) -> ActorCritic:
    """加载训练好的PPO策略"""
    obs_dim = env.observation_space.shape[0]
    action_dim = np.prod(env.action_space.shape)
    policy = ActorCritic(obs_dim, [action_dim])
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    return policy

def safe_get(dictionary: Dict, keys: str, default: float = 0.0) -> float:
    """安全获取嵌套字典值"""
    try:
        for key in keys.split('.'):
            dictionary = dictionary[key]
        return dictionary
    except (KeyError, TypeError, AttributeError):
        return default

def run_evaluation(env: ISAC_SatEnv, 
                  policy_fn: Callable, 
                  num_episodes: int = 50) -> Dict[str, List[float]]:
    """运行评估并收集四大指标"""
    results = {
        'throughput': [],
        'sensing_accuracy': [],
        'energy_consumption': [],
        'latency': [],
        'reward': [],
        'steps': []
    }
    
    # 如果是PPO策略，使用包装器
    if isinstance(policy_fn, PPOPolicyWrapper):
        ppo_wrapper = policy_fn
        policy_fn = lambda e: ppo_wrapper(e)
    else:
        ppo_wrapper = None
    
    for ep in range(num_episodes):
        obs = env.reset()
        if ppo_wrapper:
            ppo_wrapper.update_obs(obs)
        
        done = False
        episode_metrics = {
            'throughput': 0,
            'sensing_accuracy': 0,
            'energy_consumption': 0,
            'latency': 0,
            'reward': 0,
            'steps': 0
        }
        
        while not done:
            try:
                action = policy_fn(env)
                obs, reward, done, info = env.step(action)
                if ppo_wrapper:
                    ppo_wrapper.update_obs(obs)
                
                # 安全获取指标
                targets = info.get('targets', [{} for _ in range(env.config.get("num_targets", 2))])
                for t in targets:
                    episode_metrics['throughput'] += safe_get(t, 'comm.throughput')
                    episode_metrics['sensing_accuracy'] += safe_get(t, 'radar.detection_prob')
                
                episode_metrics['energy_consumption'] += np.sum(action[:, 0])  # 功率部分作为能耗
                episode_metrics['latency'] += safe_get(info, 'avg_latency')
                episode_metrics['reward'] += reward
                episode_metrics['steps'] += 1
                
            except Exception as e:
                logger.warning(f"Episode {ep+1} 评估过程中出现异常: {str(e)}")
                done = True
        
        if episode_metrics['steps'] > 0:
            # 计算平均指标
            results['throughput'].append(episode_metrics['throughput'] / episode_metrics['steps'])
            results['sensing_accuracy'].append(episode_metrics['sensing_accuracy'] / episode_metrics['steps'])
            results['energy_consumption'].append(episode_metrics['energy_consumption'] / episode_metrics['steps'])
            results['latency'].append(episode_metrics['latency'] / episode_metrics['steps'])
            results['reward'].append(episode_metrics['reward'] / episode_metrics['steps'])
            results['steps'].append(episode_metrics['steps'])
        
        logger.info(f"评估进度: {ep+1}/{num_episodes} episodes 完成")
    
    return results

def plot_comparison(results_dict: Dict[str, Dict[str, List[float]]], 
                   save_path: str = "results/comparison_plots") -> None:
    """绘制四种指标的对比图"""
    os.makedirs(save_path, exist_ok=True)
    metrics = ['throughput', 'sensing_accuracy', 'energy_consumption', 'latency']
    titles = ['通信吞吐量 (Mbps)', '感知精度 (%)', '能耗 (W)', '时延 (ms)']
    
    # 准备数据
    data = []
    for policy_name, results in results_dict.items():
        for metric in metrics:
            if metric in results and results[metric]:  # 确保指标存在且有数据
                data.append({
                    'Policy': policy_name,
                    'Metric': metric,
                    'Value': np.mean(results[metric]),
                    'Std': np.std(results[metric])
                })
    
    if not data:
        logger.error("没有有效数据可绘制")
        return
    
    df = pd.DataFrame(data)
    
    # 绘制柱状图
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        subset = df[df['Metric'] == metric]
        if not subset.empty:
            sns.barplot(x='Policy', y='Value', hue='Policy', data=subset, 
                       palette="viridis", legend=False)
            plt.errorbar(x=range(len(subset)), y=subset['Value'], yerr=subset['Std'], 
                        fmt='none', c='black', capsize=5)
            plt.title(titles[i])
            plt.ylabel('')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/metrics_comparison.png", bbox_inches='tight')
    plt.close()
    
    # 绘制雷达图
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # 准备雷达图数据
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    
    for policy_name, results in results_dict.items():
        values = []
        for m in metrics:
            val = np.mean(results[m]) if m in results and results[m] else 0
            # 归一化到0-1范围
            max_val = df[df['Metric'] == m]['Value'].max()
            norm_val = val / max_val if max_val > 0 else 0
            values.append(norm_val)
        values = np.concatenate((values, [values[0]]))  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=policy_name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_thetagrids(angles[:-1] * 180/np.pi, titles)
    ax.set_title("策略性能对比雷达图", size=20, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(f"{save_path}/radar_comparison.png", bbox_inches='tight')
    plt.close()

def main():
    """主评估函数"""
    try:
        env = ISAC_SatEnv()
        
        # 加载PPO模型
        ppo_policy = load_ppo_policy(env, "models/ppo_final.pth")
        ppo_wrapper = PPOPolicyWrapper(ppo_policy, env)
        
        # 评估三种策略
        policies = {
            "随机策略": lambda env: env.action_space.sample(),
            "贪心策略": lambda env: np.array([[0.5, 0.5]] * env.config.get("num_targets", 2), dtype=np.float32),
            "PPO策略": ppo_wrapper
        }
        
        # 运行评估
        results_dict = {}
        for name, policy_fn in policies.items():
            logger.info(f"正在评估 {name}...")
            results_dict[name] = run_evaluation(env, policy_fn, num_episodes=50)
            
            # 打印简要统计
            logger.info(f"{name} 评估结果:")
            for metric, values in results_dict[name].items():
                if values:  # 只打印有数据的指标
                    logger.info(f"  {metric}: 均值={np.mean(values):.2f} ± {np.std(values):.2f}")
        
        # 保存结果
        os.makedirs("results", exist_ok=True)
        for name, results in results_dict.items():
            pd.DataFrame(results).to_csv(f"results/{name}_metrics.csv", index=False)
        
        # 绘制对比图
        plot_comparison(results_dict)
        logger.info("评估完成，结果已保存至 results/ 目录")
    
    except Exception as e:
        logger.error(f"评估过程中发生严重错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
