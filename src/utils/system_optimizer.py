"""
Windows 11 系統優化模組
"""

import logging
import os
import sys
import psutil
import torch
from pathlib import Path


class WindowsOptimizer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.system_config = config.get('system', {})
    
    def optimize(self):
        """執行系統優化"""
        if sys.platform != "win32":
            self.logger.info("非 Windows 系統，跳過 Windows 優化")
            return
        
        self.logger.info("🔧 開始 Windows 11 系統優化...")
        
        try:
            self._set_process_priority()
            self._optimize_memory()
            self._setup_gpu_optimization()
            self._setup_cache_directories()
            self._log_system_info()
            
            self.logger.info("✅ Windows 11 系統優化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 系統優化失敗: {e}")
    
    def _set_process_priority(self):
        """設置進程優先級"""
        try:
            import psutil
            current_process = psutil.Process()
            
            # 設置為高優先級（但不是實時優先級，避免系統卡死）
            if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                self.logger.info("進程優先級已設置為高優先級")
            
        except Exception as e:
            self.logger.warning(f"設置進程優先級失敗: {e}")
    
    def _optimize_memory(self):
        """內存優化"""
        try:
            if self.system_config.get('memory_optimization', True):
                # 設置環境變量優化內存使用
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免多進程衝突
                
                # 獲取系統內存信息
                memory = psutil.virtual_memory()
                self.logger.info(f"系統內存: {memory.total // (1024**3)}GB, 可用: {memory.available // (1024**3)}GB")
                
                # 如果內存不足，建議優化
                if memory.available < 8 * (1024**3):  # 小於8GB可用內存
                    self.logger.warning("可用內存較少，建議關閉其他應用程序")
                
        except Exception as e:
            self.logger.warning(f"內存優化失敗: {e}")
    
    def _setup_gpu_optimization(self):
        """GPU 優化設置"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"檢測到 {gpu_count} 個 CUDA 設備")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    self.logger.info(f"GPU {i}: {gpu_name}, 內存: {gpu_memory // (1024**3)}GB")
                
                # 設置 GPU 內存分配策略
                gpu_memory_fraction = self.system_config.get('gpu_memory_fraction', 0.8)
                if gpu_memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
                    self.logger.info(f"GPU 內存使用限制設置為 {gpu_memory_fraction*100}%")
                
                # 啟用 CUDA 優化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
            else:
                self.logger.warning("未檢測到 CUDA 設備，將使用 CPU")
                
                # CPU 優化
                cpu_threads = self.system_config.get('cpu_threads', -1)
                if cpu_threads == -1:
                    cpu_threads = psutil.cpu_count()
                
                torch.set_num_threads(cpu_threads)
                self.logger.info(f"CPU 線程數設置為: {cpu_threads}")
                
        except Exception as e:
            self.logger.warning(f"GPU 優化設置失敗: {e}")
    
    def _setup_cache_directories(self):
        """設置緩存目錄"""
        try:
            cache_dir = Path(self.system_config.get('cache_dir', './cache'))
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 設置 Transformers 緩存目錄
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / 'transformers')
            os.environ['HF_HOME'] = str(cache_dir / 'huggingface')
            
            # 設置 Torch 緩存目錄
            os.environ['TORCH_HOME'] = str(cache_dir / 'torch')
            
            self.logger.info(f"緩存目錄設置為: {cache_dir}")
            
        except Exception as e:
            self.logger.warning(f"設置緩存目錄失敗: {e}")
    
    def _log_system_info(self):
        """記錄系統信息"""
        try:
            # CPU 信息
            cpu_info = {
                'processor': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown'
            }
            
            # 內存信息
            memory = psutil.virtual_memory()
            memory_info = {
                'total': f"{memory.total // (1024**3)}GB",
                'available': f"{memory.available // (1024**3)}GB",
                'used_percent': f"{memory.percent}%"
            }
            
            # 磁盤信息
            disk = psutil.disk_usage('.')
            disk_info = {
                'total': f"{disk.total // (1024**3)}GB",
                'free': f"{disk.free // (1024**3)}GB",
                'used_percent': f"{(disk.used / disk.total) * 100:.1f}%"
            }
            
            self.logger.info("系統信息:")
            self.logger.info(f"  CPU: {cpu_info['processor']}核心/{cpu_info['logical_cores']}線程 @ {cpu_info['frequency']}MHz")
            self.logger.info(f"  內存: {memory_info['used_percent']} 已使用 ({memory_info['available']} 可用 / {memory_info['total']} 總計)")
            self.logger.info(f"  磁盤: {disk_info['used_percent']} 已使用 ({disk_info['free']} 可用 / {disk_info['total']} 總計)")
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                self.logger.info(f"  GPU: {gpu_name} ({gpu_memory}GB)")
            
        except Exception as e:
            self.logger.warning(f"獲取系統信息失敗: {e}")