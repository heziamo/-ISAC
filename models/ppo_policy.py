# models/ppo_policy.py
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np

class ActorCritic(nn.Module):
    """PPO 策略网络"""
    
    def __init__(self, obs_dim, action_dims, hidden_size=256):
        """
        初始化 ActorCritic 网络
        
        参数:
            obs_dim (tuple): 观测空间的维度
            action_dims (list): 每个动作空间的维度 [power_dim, subcarrier_dim, offload_dim]
            hidden_size (int): 隐藏层大小
        """
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # 连续动作头
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_dims[0])
        )

        # 评论家网络 (价值)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, obs):
        """
        前向传播
        
        参数:
            obs (torch.Tensor): 观测值
        
        返回:
            power_mu (torch.Tensor): 功率动作的均值
            subcarrier_logits (torch.Tensor): 子载波动作的logits
            offload_mu (torch.Tensor): 卸载比例动作的均值
            value (torch.Tensor): 状态价值
        """
        features = self.feature_extractor(obs)
        
        # 演员输出
        action_mu = self.actor(features)
        
        # 评论家输出
        value = self.critic(features)
        
        return action_mu, value
    
    def get_action(self, obs):
        """
        根据当前观测获取动作
        
        参数:
            obs (numpy.ndarray): 环境观测值
        
        返回:
            action (tuple): (功率, 子载波, 卸载比例)
            log_prob (torch.Tensor): 动作的对数概率
            value (torch.Tensor): 状态价值
        """
        with torch.no_grad():
            # 转换为张量
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # 前向传播
            action_mu, value = self.forward(obs_tensor)
            
            # 动作分布 (连续)
            dist = Normal(action_mu, 0.1)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action.squeeze(0).numpy(), log_prob, value

class PPO:
    """PPO 算法实现"""
    
    def __init__(self, 
                 policy, 
                 optimizer, 
                 clip_epsilon=0.2, 
                 value_coeff=0.5, 
                 entropy_coeff=0.01):
        """
        初始化 PPO
        
        参数:
            policy (ActorCritic): 策略网络
            optimizer (torch.optim.Optimizer): 优化器
            clip_epsilon (float): PPO 裁剪参数
            value_coeff (float): 价值损失系数
            entropy_coeff (float): 熵正则化系数
        """
        self.policy = policy
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
    
    def update(self, rollouts):
        """
        使用收集的经验更新策略
        
        参数:
            rollouts (dict): 包含经验数据的字典
        """
        # 从经验中提取数据
        obs = torch.as_tensor(rollouts['obs'], dtype=torch.float32)
        actions = torch.as_tensor(rollouts['actions'], dtype=torch.float32)
        old_log_probs = torch.as_tensor(rollouts['log_probs'], dtype=torch.float32)
        returns = torch.as_tensor(rollouts['returns'], dtype=torch.float32)
        advantages = torch.as_tensor(rollouts['advantages'], dtype=torch.float32)
        
        # 前向传播获取新策略的值
        action_mu, values = self.policy(obs)
        
        # 计算新策略的对数概率
        dist = Normal(action_mu, 0.1)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # 计算熵
        entropy = dist.entropy().mean()
        
        # 计算概率比
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO 裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
        
        # 总损失
        loss = actor_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        
        }
