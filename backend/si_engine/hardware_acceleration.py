"""
HardwareAcceleration - Hardware detection and optimization for SI engine
Detects available compute resources and optimizes SI operations
"""

import platform
import psutil
import multiprocessing
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    """Hardware configuration information"""
    cpu_count: int
    cpu_freq_mhz: float
    ram_total_gb: float
    ram_available_gb: float
    platform: str
    architecture: str
    has_gpu: bool = False
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'cpu_count': self.cpu_count,
            'cpu_freq_mhz': self.cpu_freq_mhz,
            'ram_total_gb': self.ram_total_gb,
            'ram_available_gb': self.ram_available_gb,
            'platform': self.platform,
            'architecture': self.architecture,
            'has_gpu': self.has_gpu,
            'gpu_info': self.gpu_info
        }

class HardwareAccelerator:
    """
    Hardware detection and acceleration manager
    Optimizes SI engine based on available hardware
    """
    
    def __init__(self):
        self.hardware_info = self.detect_hardware()
        self.optimal_workers = self._calculate_optimal_workers()
        logger.info(f"Hardware detected: {self.hardware_info.cpu_count} CPUs, "
                   f"{self.hardware_info.ram_total_gb:.1f}GB RAM")
    
    def detect_hardware(self) -> HardwareInfo:
        """
        Detect available hardware resources
        """
        cpu_count = multiprocessing.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
        
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024 ** 3)
        ram_available_gb = memory.available / (1024 ** 3)
        
        sys_platform = platform.system()
        architecture = platform.machine()
        
        # Try to detect GPU
        has_gpu, gpu_info = self._detect_gpu()
        
        return HardwareInfo(
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq_mhz,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            platform=sys_platform,
            architecture=architecture,
            has_gpu=has_gpu,
            gpu_info=gpu_info
        )
    
    def _detect_gpu(self) -> tuple[bool, Dict[str, Any]]:
        """Detect GPU availability and information"""
        gpu_info = {}
        
        # Try CUDA (NVIDIA)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['type'] = 'CUDA'
                gpu_info['name'] = torch.cuda.get_device_name(0)
                gpu_info['count'] = torch.cuda.device_count()
                gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return True, gpu_info
        except ImportError:
            pass
        
        # Try to detect via other methods
        try:
            # Check for common GPU libraries
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(',')
                    gpu_info['type'] = 'NVIDIA'
                    gpu_info['name'] = parts[0].strip()
                    if len(parts) > 1:
                        gpu_info['memory'] = parts[1].strip()
                    return True, gpu_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return False, gpu_info
    
    def _calculate_optimal_workers(self) -> int:
        """
        Calculate optimal number of workers for parallel processing
        Based on available CPU cores and memory
        """
        cpu_count = self.hardware_info.cpu_count
        available_ram_gb = self.hardware_info.ram_available_gb
        
        # Use 75% of CPUs to leave some for system
        optimal_by_cpu = max(1, int(cpu_count * 0.75))
        
        # Estimate 1 worker per 2GB available RAM
        optimal_by_ram = max(1, int(available_ram_gb / 2))
        
        # Take the minimum to avoid overload
        return min(optimal_by_cpu, optimal_by_ram)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get current system resource usage
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024 ** 3),
            'memory_used_gb': memory.used / (1024 ** 3),
            'timestamp': time.time()
        }
    
    def should_throttle(self, cpu_threshold: float = 90.0, 
                       memory_threshold: float = 90.0) -> tuple[bool, str]:
        """
        Check if system resources are high and processing should be throttled
        Returns (should_throttle, reason)
        """
        stats = self.get_system_stats()
        
        if stats['cpu_percent'] > cpu_threshold:
            return True, f"High CPU usage: {stats['cpu_percent']:.1f}%"
        
        if stats['memory_percent'] > memory_threshold:
            return True, f"High memory usage: {stats['memory_percent']:.1f}%"
        
        return False, "Resources OK"
    
    def optimize_batch_size(self, default_batch_size: int, 
                           memory_per_item_mb: float = 10.0) -> int:
        """
        Calculate optimal batch size based on available memory
        """
        available_mb = self.hardware_info.ram_available_gb * 1024
        
        # Use 50% of available memory for batching
        safe_memory_mb = available_mb * 0.5
        optimal_batch = int(safe_memory_mb / memory_per_item_mb)
        
        # Ensure it's at least 1 and not more than default * 2
        return max(1, min(optimal_batch, default_batch_size * 2))
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """
        Get recommended parallel processing configuration
        """
        return {
            'max_workers': self.optimal_workers,
            'use_threading': self.hardware_info.cpu_count > 1,
            'chunk_size': max(1, self.optimal_workers * 2),
            'hardware_info': self.hardware_info.to_dict()
        }
    
    def accelerate_pattern_matching(self, use_parallel: bool = True) -> Dict[str, Any]:
        """
        Get acceleration settings for pattern matching operations
        """
        config = {
            'use_parallel': use_parallel and self.hardware_info.cpu_count > 1,
            'num_workers': self.optimal_workers if use_parallel else 1,
            'batch_size': self.optimize_batch_size(100, memory_per_item_mb=5.0)
        }
        
        if self.hardware_info.has_gpu:
            config['use_gpu'] = True
            config['gpu_batch_size'] = self.optimize_batch_size(500, memory_per_item_mb=2.0)
        
        return config