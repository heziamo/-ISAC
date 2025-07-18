"""
plot_metrics.py - ISAC基线策略可视化工具
从results目录读取CSV结果文件，生成性能对比图
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import unittest
import coverage

class BaselineVisualizer:
    def __init__(self):
        # 初始化路径
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results"
        self.output_dir = self.base_dir / "report/figures"
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 定义预期的CSV文件名
        self.expected_files = {
            'greedy': 'greedy_baseline.csv',
            'random': 'random_baseline.csv'
        }

    def validate_files(self) -> bool:
        """验证结果文件是否存在"""
        missing_files = [
            f for f in self.expected_files.values() 
            if not (self.results_dir / f).exists()
        ]
        if missing_files:
            print(f"错误：缺少结果文件: {missing_files}")
            return False
        return True

    def load_csv_data(self, policy_type: str) -> list:
        """加载指定策略的CSV数据"""
        filename = self.expected_files[policy_type]
        filepath = self.results_dir / filename
        
        data = []
        with open(filepath, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值类型
                converted = {
                    'episode': int(row['episode']),
                    'total_reward': float(row['total_reward']),
                    'steps': int(row['steps']),
                    'avg_comm_snr': float(row['avg_comm_snr']),
                    'avg_radar_snr': float(row['avg_radar_snr']),
                    'comm_success_rate': float(row['comm_success_rate']),
                    'radar_success_rate': float(row['radar_success_rate'])
                }
                data.append(converted)
        return data

    def calculate_metrics(self, data: list) -> dict:
        """计算关键指标统计量"""
        if not data:
            return {}
            
        metrics = {
            'avg_reward': np.mean([d['total_reward'] for d in data]),
            'max_reward': np.max([d['total_reward'] for d in data]),
            'min_reward': np.min([d['total_reward'] for d in data]),
            'avg_comm_snr': np.mean([d['avg_comm_snr'] for d in data]),
            'avg_radar_snr': np.mean([d['avg_radar_snr'] for d in data]),
            'comm_success': np.mean([d['comm_success_rate'] for d in data]),
            'radar_success': np.mean([d['radar_success_rate'] for d in data])
        }
        return metrics

    def plot_comparison(self, greedy_metrics: dict, random_metrics: dict):
        """生成对比柱状图"""
        # 准备数据
        labels = ['总奖励', '通信SNR', '雷达SNR', '通信成功率', '雷达成功率']
        greedy_values = [
            greedy_metrics['avg_reward'],
            greedy_metrics['avg_comm_snr'],
            greedy_metrics['avg_radar_snr'],
            greedy_metrics['comm_success'],
            greedy_metrics['radar_success']
        ]
        random_values = [
            random_metrics['avg_reward'],
            random_metrics['avg_comm_snr'],
            random_metrics['avg_radar_snr'],
            random_metrics['comm_success'],
            random_metrics['radar_success']
        ]
        
        # 创建图表
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, greedy_values, width, label='贪婪策略')
        rects2 = ax.bar(x + width/2, random_values, width, label='随机策略')
        
        # 添加标签和标题
        ax.set_ylabel('性能值')
        ax.set_title('基线策略性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        
        # 保存图像
        output_path = self.output_dir / "fig_baseline.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

    def generate_report(self):
        """生成完整性能报告"""
        if not self.validate_files():
            return None
            
        # 加载数据
        greedy_data = self.load_csv_data('greedy')
        random_data = self.load_csv_data('random')
        
        # 计算指标
        greedy_metrics = self.calculate_metrics(greedy_data)
        random_metrics = self.calculate_metrics(random_data)
        
        # 生成图表
        plot_path = self.plot_comparison(greedy_metrics, random_metrics)
        
        print(f"可视化结果已保存至: {plot_path}")
        return {
            'greedy': greedy_metrics,
            'random': random_metrics,
            'plot_path': str(plot_path)
        }

# 单元测试
class TestVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 创建测试用CSV文件
        test_dir = Path(__file__).parent / "test_results"
        os.makedirs(test_dir, exist_ok=True)
        
        # 创建模拟的贪婪策略结果
        with open(test_dir / "greedy_baseline.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'steps',
                'avg_comm_snr', 'avg_radar_snr',
                'comm_success_rate', 'radar_success_rate'
            ])
            writer.writerow([
                1, 100.5, 10, 25.3, 30.2, 0.9, 0.8
            ])
        
        # 创建模拟的随机策略结果
        with open(test_dir / "random_baseline.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'steps',
                'avg_comm_snr', 'avg_radar_snr',
                'comm_success_rate', 'radar_success_rate'
            ])
            writer.writerow([
                1, 50.2, 10, 15.1, 20.3, 0.6, 0.5
            ])

    def setUp(self):
        self.visualizer = BaselineVisualizer()
        self.visualizer.results_dir = Path(__file__).parent / "test_results"

    def test_file_validation(self):
        """测试文件验证逻辑"""
        self.assertTrue(self.visualizer.validate_files())

    def test_csv_loading(self):
        """测试CSV加载功能"""
        data = self.visualizer.load_csv_data('greedy')
        self.assertEqual(len(data), 1)
        self.assertAlmostEqual(data[0]['total_reward'], 100.5)

    def test_metrics_calculation(self):
        """测试指标计算"""
        test_data = [{
            'total_reward': 100,
            'avg_comm_snr': 20,
            'avg_radar_snr': 30,
            'comm_success_rate': 0.8,
            'radar_success_rate': 0.7
        }]
        metrics = self.visualizer.calculate_metrics(test_data)
        self.assertAlmostEqual(metrics['avg_reward'], 100)
        self.assertAlmostEqual(metrics['comm_success'], 0.8)

    def test_plot_generation(self):
        """测试图表生成"""
        greedy_metrics = {
            'avg_reward': 100,
            'avg_comm_snr': 20,
            'avg_radar_snr': 30,
            'comm_success': 0.8,
            'radar_success': 0.7
        }
        random_metrics = {
            'avg_reward': 50,
            'avg_comm_snr': 15,
            'avg_radar_snr': 25,
            'comm_success': 0.6,
            'radar_success': 0.5
        }
        plot_path = self.visualizer.plot_comparison(greedy_metrics, random_metrics)
        self.assertTrue(os.path.exists(plot_path))

if __name__ == "__main__":
    # 配置覆盖率检测
    cov = coverage.Coverage(
        source=[os.path.dirname(__file__)],
        omit=["*test*"]
    )
    cov.start()
    
    # 运行单元测试
    unittest.main(exit=False)
    
    # 生成覆盖率报告
    cov.stop()
    cov.save()
    cov.xml_report(outfile="coverage.xml")
    print("\n测试覆盖率报告已生成: coverage.xml")
    
    # 执行可视化任务
    if not unittest.result.TestResult().wasSuccessful():
        print("警告: 存在测试失败情况，可视化结果可能不准确")
    
    visualizer = BaselineVisualizer()
    report = visualizer.generate_report()
    
    if report:
        print("\n性能对比结果:")
        print(f"贪婪策略平均奖励: {report['greedy']['avg_reward']:.2f}")
        print(f"随机策略平均奖励: {report['random']['avg_reward']:.2f}")