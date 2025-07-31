# train/train_ppo.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.ppo_policy import ActorCritic, PPO
from isac_sat_env import ISAC_SatEnv
from collections import deque
import argparse
import wandb

class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """计算GAE优势函数和回报"""
        returns = []
        advantages = []
        gae = 0
        next_value = last_value
        next_done = 0
        
        # 反向计算
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t+1]
                next_values = self.values[t+1]
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'returns': np.array(returns),
            'advantages': np.array(advantages)
        }

def train_ppo(config=None):
    """PPO训练主函数"""
    # 初始化环境
    env = ISAC_SatEnv(config)
    num_targets = env.config.get("num_targets", 2)
    obs_dim = env.observation_space.shape[0]
    action_dim = np.prod(env.action_space.shape)  # 修正动作维度计算

    # 创建策略网络（只用一个连续动作头）
    policy = ActorCritic(obs_dim, [action_dim])
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    ppo = PPO(policy, optimizer)
    
    # 初始化WandB
    wandb.init(project="satellite-isac-ppo", config=config)
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir='runs/ppo_train')
    
    # 训练参数
    num_episodes = 1000
    max_steps = 200
    batch_size = 64
    update_frequency = 5  # 每5个episode更新一次
    
    # 训练统计
    episode_rewards = []
    best_reward = -np.inf
    
    # 创建缓冲区
    buffer = RolloutBuffer()
    
    # 训练循环
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        buffer.reset()
        fairness_comm_list = []
        fairness_radar_list = []

        for step in range(max_steps):
            # 获取动作
            action, log_prob, value = policy.get_action(obs)
            # 动作reshape
            action = action.reshape((num_targets, 2))
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 存储经验
            buffer.add(obs, action.flatten(), log_prob, value, reward, done)
            
            # 更新状态
            obs = next_obs
            episode_reward += reward
            fairness_comm_list.append(info.get('fairness_comm', 0))
            fairness_radar_list.append(info.get('fairness_radar', 0))
            if done:
                break

        # 记录奖励
        episode_rewards.append(episode_reward)
        wandb.log({
            'episode_reward': episode_reward,
            'fairness_comm': np.mean(fairness_comm_list),
            'fairness_radar': np.mean(fairness_radar_list)
        }, step=episode)
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Fairness/Comm', np.mean(fairness_comm_list), episode)
        writer.add_scalar('Fairness/Radar', np.mean(fairness_radar_list), episode)
        
        # 定期更新策略
        if episode % update_frequency == 0:
            # 计算最终状态的价值
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, last_value = policy(obs_tensor)
                last_value = last_value.item()
            
            # 计算回报和优势
            rollouts = buffer.compute_returns(last_value)
            
            # 更新策略
            loss_info = ppo.update(rollouts)
            
            # 记录损失
            wandb.log({
                'loss/total': loss_info['total_loss'],
                'loss/actor': loss_info['actor_loss'],
                'loss/value': loss_info['value_loss'],
                'entropy': loss_info['entropy']
            }, step=episode)
            
            print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                  f"Loss={loss_info['total_loss']:.4f}")
        
        # 定期保存模型
        if episode % 50 == 0 and episode > 0:
            # 检查是否为最佳模型
            avg_reward = np.mean(episode_rewards[-10:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(policy.state_dict(), f'models/ppo_best_{episode}.pth')
                print(f"Saved best model at episode {episode} with avg reward {avg_reward:.2f}")
    
    # 保存最终模型
    torch.save(policy.state_dict(), 'models/ppo_final.pth')
    writer.close()
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for Satellite ISAC')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    args = parser.parse_args()
    
    # 加载配置（如果有）
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    train_ppo(config)