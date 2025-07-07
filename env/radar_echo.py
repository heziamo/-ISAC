import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light, pi
from scipy.signal import fftconvolve

class RadarEchoModel:
    def __init__(self, fc=10e9,  # 载频 10GHz
                 bw=30e6,  # 带宽 30MHz
                 prf=1000,  # 脉冲重复频率 (Hz)
                 pulse_width=10e-6,  # 脉冲宽度 (s)
                 n_pulses=64,  # 脉冲数
                 fs=100e6,  # 采样率 (Hz)
                 noise_power=1e-6):  # 噪声功率
        
        self.fc = fc  # 载频
        self.bw = bw  # 带宽
        self.prf = prf  # 脉冲重复频率
        self.pulse_width = pulse_width  # 脉冲宽度
        self.n_pulses = n_pulses  # 脉冲数
        self.fs = fs  # 采样率
        self.noise_power = noise_power  # 噪声功率
        
        # 计算波长
        self.lambda_ = speed_of_light / self.fc
        
        # 计算时间参数
        self.pri = 1 / self.prf  # 脉冲重复间隔
        self.n_samples_per_pulse = int(self.pulse_width * self.fs)
        self.n_samples_per_pri = int(self.pri * self.fs)
        
        # 生成线性调频信号
        self.chirp = self._generate_chirp()
        
    def _generate_chirp(self):
        """生成线性调频信号"""
        t = np.linspace(0, self.pulse_width, self.n_samples_per_pulse, endpoint=False)
        chirp = np.exp(1j * pi * (self.bw / self.pulse_width) * t**2)
        return chirp
    
    def _range_profile(self, targets):
        """生成距离像"""
        # 初始化回波信号
        echo = np.zeros(self.n_samples_per_pri, dtype=np.complex128)
        
        # 对每个目标添加回波
        for rcs, range_, vel in targets:
            # 计算时延 (秒)
            delay = 2 * range_ / speed_of_light
            
            # 计算多普勒相位变化
            doppler_phase = 2 * pi * 2 * vel / self.lambda_ * self.pri
            
            # 计算时延对应的采样点
            delay_samples = int(delay * self.fs)
            
            # 确保时延在合理范围内
            if delay_samples + len(self.chirp) > len(echo):
                continue
                
            # 添加目标回波
            echo[delay_samples:delay_samples+len(self.chirp)] += (
                np.sqrt(rcs) * self.chirp * np.exp(1j * doppler_phase))
        
        return echo
    
    def _add_clutter(self, echo, clutter_rcs=1e3, clutter_range=[5000, 10000]):
        """添加杂波"""
        # 生成随机杂波
        n_clutter = 100  # 杂波点数
        ranges = np.random.uniform(clutter_range[0], clutter_range[1], n_clutter)
        rcs_values = clutter_rcs * np.random.rand(n_clutter)
        
        for rcs, range_ in zip(rcs_values, ranges):
            delay = 2 * range_ / speed_of_light
            delay_samples = int(delay * self.fs)
            
            if delay_samples + len(self.chirp) > len(echo):
                continue
                
            echo[delay_samples:delay_samples+len(self.chirp)] += (
                np.sqrt(rcs) * self.chirp * np.random.randn() * np.exp(1j * 2 * pi * np.random.rand()))
        
        return echo
    
    def _add_noise(self, echo):
        """添加噪声"""
        noise = np.sqrt(self.noise_power/2) * (
            np.random.randn(len(echo)) + 1j * np.random.randn(len(echo)))
        return echo + noise
    
    def generate_echo(self, targets, add_clutter=True, add_noise=True):
        """生成雷达回波信号"""
        # 初始化回波矩阵 (距离门 x 脉冲数)
        echo_matrix = np.zeros((self.n_samples_per_pri, self.n_pulses), dtype=np.complex128)
        
        for pulse_idx in range(self.n_pulses):
            # 生成当前脉冲的回波
            echo = self._range_profile(targets)
            
            # 更新目标位置 (考虑速度)
            targets = [(rcs, range_ + vel * self.pri, vel) for rcs, range_, vel in targets]
            
            # 添加杂波
            if add_clutter:
                echo = self._add_clutter(echo)
                
            # 添加噪声
            if add_noise:
                echo = self._add_noise(echo)
                
            # 存储回波
            echo_matrix[:, pulse_idx] = echo
        
        return echo_matrix
    
    def pulse_compression(self, echo_matrix):
        """脉冲压缩 (匹配滤波)"""
        compressed = np.zeros_like(echo_matrix)
        for i in range(echo_matrix.shape[1]):
            compressed[:, i] = fftconvolve(echo_matrix[:, i], np.conj(self.chirp[::-1]), mode='same')
        return compressed
    
    def doppler_processing(self, compressed_matrix):
        """多普勒处理 (FFT)"""
        return np.fft.fft(compressed_matrix, axis=1)