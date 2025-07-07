import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light

class SatelliteChannel:
    def __init__(self, frequency=12e9,  # 载波频率 (Hz)
                 satellite_height=35786e3,  # 卫星高度 (m) GEO轨道
                 elevation_angle=30,  # 仰角 (度)
                 tx_power=10,  # 发射功率 (dBW)
                 tx_gain=30,  # 发射天线增益 (dBi)
                 rx_gain=30,  # 接收天线增益 (dBi)
                 bandwidth=36e6,  # 带宽 (Hz)
                 temperature=290):  # 系统噪声温度 (K)
        
        self.frequency = frequency
        self.satellite_height = satellite_height
        self.elevation_angle = np.radians(elevation_angle)
        self.tx_power = tx_power
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.bandwidth = bandwidth
        self.temperature = temperature
        self.boltzmann = 1.38064852e-23  # 玻尔兹曼常数
        
        # 计算波长
        self.wavelength = speed_of_light / self.frequency
        
    def calculate_distance(self):
        """计算卫星与地面站之间的距离"""
        R_earth = 6371e3  # 地球半径 (m)
        # 使用余弦定理计算斜距
        distance = np.sqrt(R_earth**2 + (R_earth + self.satellite_height)**2 - 
                   2 * R_earth * (R_earth + self.satellite_height) * 
                   np.cos(np.pi/2 + self.elevation_angle))
        return distance
    
    def free_space_path_loss(self):
        """计算自由空间路径损耗"""
        distance = self.calculate_distance()
        fspl = 20 * np.log10(distance) + 20 * np.log10(self.frequency) + 20 * np.log10(4 * np.pi / speed_of_light)
        return fspl
    
    def atmospheric_loss(self):
        """计算大气衰减 """
        # 对于Ku波段(12GHz)，典型值为0.5-1.5dB
        return 1.0  # dB
    
    def rain_attenuation(self, rain_rate):
        """计算雨衰 """
        # rain_rate in mm/h
        if self.frequency < 10e9:
            return 0.0
        elif self.frequency < 20e9:
            return 0.01 * rain_rate  # dB
        else:
            return 0.03 * rain_rate  # dB
    
    def doppler_shift(self, relative_velocity):
        """计算多普勒频移"""
        return (relative_velocity / speed_of_light) * self.frequency
    
    def received_power(self, rain_rate=0):
        """计算接收功率"""
        fspl = self.free_space_path_loss()
        atm_loss = self.atmospheric_loss()
        rain_loss = self.rain_attenuation(rain_rate)
        
        # EIRP (等效全向辐射功率)
        eirp = self.tx_power + self.tx_gain
        
        # 接收功率 (dBW)
        pr = eirp - fspl - atm_loss - rain_loss + self.rx_gain
        return pr
    
    def snr(self, rain_rate=0):
        """计算信噪比(SNR)"""
        pr = self.received_power(rain_rate)
        
        # 噪声功率 (dBW)
        noise_power = 10 * np.log10(self.boltzmann * self.temperature * self.bandwidth)
        
        # SNR (dB)
        snr_value = pr - noise_power
        return snr_value
    
    def capacity(self, rain_rate=0):
        """计算信道容量 (Shannon公式)"""
        snr_linear = 10 ** (self.snr(rain_rate) / 10)
        capacity = self.bandwidth * np.log2(1 + snr_linear)
        return capacity / 1e6  # 转换为Mbps