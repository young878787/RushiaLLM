"""
GPU 資源管理模組
負責多GPU智能分配、記憶體管理和設備優化
"""

import logging
import torch
import numpy as np
import datetime
import gc
import os
from typing import Optional, List, Dict, Any


class MultiGPUManager:
    """多GPU管理器 - 智能分配和管理多張GPU卡"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_gpus = []
        self.gpu_info = {}
        self.device_map = {}
        
        # 🔥 新增：日誌控制機制
        self._status_logged = False  # 追蹤是否已經顯示過狀態
        self._last_log_time = None   # 最後一次記錄時間
        self._log_interval = 300     # 最小間隔時間（秒）
        
        self._initialize_gpus()
    
    def _initialize_gpus(self):
        """初始化GPU信息"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA 不可用，將使用 CPU")
            return
        
        gpu_count = torch.cuda.device_count()
        self.logger.info(f"檢測到 {gpu_count} 張GPU卡")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'id': i,
                'name': props.name,
                'total_memory': props.total_memory,
                'total_memory_gb': props.total_memory / (1024**3),
                'available': True,
                'allocated_memory': 0,
                'reserved_memory': 0
            }
            
            self.available_gpus.append(i)
            self.gpu_info[i] = gpu_info
            
            self.logger.info(f"GPU {i}: {props.name} - {gpu_info['total_memory_gb']:.1f}GB")
    
    def get_optimal_device_map(self, model_type: str = "llm") -> Dict[str, Any]:
        """獲取最佳的設備映射策略 - 混合精度優化版"""
        if not self.available_gpus:
            return None
        
        if model_type == "llm":
            # 🔥 LLM主模型：8bit量化，統一使用4張卡配置
            if len(self.available_gpus) >= 4:
                primary_gpus = self.available_gpus[:4]
                
                self.logger.info(f"🚀 LLM 8bit量化模型將使用GPU {primary_gpus} (Auto分配模式)")
                
                # 🔧 統一記憶體分配策略：4張卡，每張4GB
                return {
                    "auto_mode": True,
                    "allowed_devices": primary_gpus,
                    "max_memory": {i: "4GB" for i in primary_gpus},
                    "device_map_strategy": "balanced"
                }
            else:
                # GPU不足4張，使用標準auto，8bit量化記憶體效率高
                self.logger.info("🔧 GPU數量不足4張，8bit量化模型記憶體效率高")
                self.logger.info("💡 8bit量化可以在更少GPU上運行")
                return "auto"
        
        elif model_type == "embedding":
            # 🔥 嵌入模型：FP16版本，使用第5張卡（基於4張LLM卡之後）
            if len(self.available_gpus) >= 5:
                embedding_gpu = self.available_gpus[4]  # 使用第5張卡
                self.logger.info(f"🎯 Embedding FP16模型將使用GPU {embedding_gpu}")
                return embedding_gpu
            elif len(self.available_gpus) > 1:
                # 如果有多張卡但不足5張，使用最後一張
                embedding_gpu = self.available_gpus[-1]
                self.logger.info(f"🎯 Embedding FP16模型將使用GPU {embedding_gpu} (最後一張卡)")
                return embedding_gpu
            else:
                # 只有1張卡，共享使用，8bit+FP16混合配置下記憶體壓力小
                self.logger.info("🔧 只有1張GPU，8bit LLM + FP16 Embedding共享使用")
                self.logger.info("💡 8bit量化降低記憶體壓力，共享使用可行")
                return 0
        
        return "auto"
    
    def get_memory_allocation_config(self, model_type: str = "llm") -> Dict[str, Any]:
        """統一的記憶體分配配置方法"""
        device_map_result = self.get_optimal_device_map(model_type)
        
        if model_type == "llm":
            if isinstance(device_map_result, dict) and device_map_result.get("auto_mode"):
                # 返回LLM模型的完整配置
                return {
                    "use_device_map": True,
                    "device_map": "auto",
                    "max_memory": device_map_result["max_memory"],
                    "allowed_devices": device_map_result["allowed_devices"],
                    "memory_per_gpu": "4GB",
                    "total_gpus": len(device_map_result["allowed_devices"])
                }
            else:
                # 標準auto配置
                return {
                    "use_device_map": True,
                    "device_map": "auto",
                    "max_memory": None,
                    "allowed_devices": self.available_gpus,
                    "memory_per_gpu": "auto",
                    "total_gpus": len(self.available_gpus)
                }
        
        elif model_type == "embedding":
            if isinstance(device_map_result, int):
                # 指定GPU
                return {
                    "use_device_map": True,
                    "device_map": {"": device_map_result},
                    "target_gpu": device_map_result,
                    "device": f"cuda:{device_map_result}"
                }
            else:
                # 回退配置
                return {
                    "use_device_map": True,
                    "device_map": "auto",
                    "target_gpu": None,
                    "device": "auto"
                }
        
        return {}
    
    def log_memory_allocation_info(self, model_type: str = "llm"):
        """記錄記憶體分配信息"""
        config = self.get_memory_allocation_config(model_type)
        
        if model_type == "llm":
            if config.get("max_memory"):
                self.logger.info("📊 LLM記憶體分配策略:")
                self.logger.info(f"   使用GPU: {config['allowed_devices']}")
                self.logger.info(f"   每張卡記憶體: {config['memory_per_gpu']}")
                self.logger.info(f"   總記憶體分配: {config['total_gpus']} × {config['memory_per_gpu']} = {float(config['memory_per_gpu'].rstrip('GB')) * config['total_gpus']}GB")
            else:
                self.logger.info("📊 LLM使用標準Auto分配策略")
        
        elif model_type == "embedding":
            if config.get("target_gpu") is not None:
                self.logger.info(f"📊 Embedding分配到GPU {config['target_gpu']}")
            else:
                self.logger.info("📊 Embedding使用Auto分配策略")
    
    def get_gpu_memory_info(self) -> Dict[int, Dict[str, float]]:
        """獲取所有GPU的記憶體使用情況"""
        memory_info = {}
        
        for gpu_id in self.available_gpus:
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            total = self.gpu_info[gpu_id]['total_memory_gb']
            
            memory_info[gpu_id] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved,
                'utilization': (reserved / total) * 100
            }
            
            self.gpu_info[gpu_id]['allocated_memory'] = allocated
            self.gpu_info[gpu_id]['reserved_memory'] = reserved
        
        return memory_info
    
    def clear_gpu_memory(self, gpu_ids: Optional[List[int]] = None):
        """清理指定GPU的記憶體"""
        if gpu_ids is None:
            gpu_ids = self.available_gpus
        
        for gpu_id in gpu_ids:
            if gpu_id in self.available_gpus:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
        
        gc.collect()
        self.logger.info(f"已清理GPU {gpu_ids} 的記憶體")
    
    def get_total_gpu_memory(self) -> float:
        """獲取總GPU記憶體"""
        return sum(info['total_memory_gb'] for info in self.gpu_info.values())
    
    def get_available_memory(self) -> float:
        """獲取可用GPU記憶體"""
        memory_info = self.get_gpu_memory_info()
        return sum(info['free_gb'] for info in memory_info.values())
    
    def diagnose_gpu_allocation(self, llm_model=None) -> Dict[str, Any]:
        """診斷GPU分配狀況 - 專門用於排查設備不同步問題"""
        diagnosis = {
            "timestamp": str(datetime.datetime.now()),
            "device_allocation": {},
            "tensor_devices": {},
            "model_distribution": {},
            "potential_issues": []
        }
        
        try:
            # 1. 檢查模型設備分佈
            if llm_model and hasattr(llm_model, 'hf_device_map'):
                device_distribution = {}
                for module_name, device in llm_model.hf_device_map.items():
                    device_str = f"cuda:{device}" if isinstance(device, int) else str(device)
                    if device_str not in device_distribution:
                        device_distribution[device_str] = []
                    device_distribution[device_str].append(module_name)
                
                diagnosis["model_distribution"] = device_distribution
                
                # 檢查是否有設備分佈不均
                device_counts = {device: len(modules) for device, modules in device_distribution.items()}
                if len(device_counts) > 1:
                    max_modules = max(device_counts.values())
                    min_modules = min(device_counts.values())
                    if max_modules - min_modules > 5:  # 模組數量差異過大
                        diagnosis["potential_issues"].append(
                            f"設備分佈不均：{device_counts}"
                        )
            
            # 2. 檢查記憶體使用
            memory_info = self.get_gpu_memory_info()
            diagnosis["memory_usage"] = memory_info
            
            # 檢查記憶體使用不均
            if len(memory_info) > 1:
                utilizations = [info['utilization'] for info in memory_info.values()]
                max_util = max(utilizations)
                min_util = min(utilizations)
                if max_util - min_util > 30:  # 使用率差異超過30%
                    diagnosis["potential_issues"].append(
                        f"記憶體使用不均：最高{max_util:.1f}% 最低{min_util:.1f}%"
                    )
            
            # 3. 檢查設備可見性
            diagnosis["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "未設定")
            diagnosis["available_gpus"] = self.available_gpus
            
            return diagnosis
            
        except Exception as e:
            diagnosis["error"] = str(e)
            return diagnosis
    
    def log_gpu_status(self, force: bool = False, reason: str = ""):
        """記錄GPU狀態 - 優化版，避免重複顯示"""
        import time
        
        current_time = time.time()
        
        # 🔥 控制日誌顯示頻率
        if not force:
            # 如果已經顯示過且未超過間隔時間，則跳過
            if (self._status_logged and 
                self._last_log_time and 
                current_time - self._last_log_time < self._log_interval):
                return
        
        memory_info = self.get_gpu_memory_info()
        total_memory = self.get_total_gpu_memory()
        available_memory = self.get_available_memory()
        
        # 🔥 簡化的狀態報告
        if not self._status_logged:
            # 第一次顯示時，顯示完整信息
            self.logger.info("=== GPU 狀態報告 ===")
            self.logger.info(f"總GPU記憶體: {total_memory:.1f}GB")
            self.logger.info(f"可用記憶體: {available_memory:.1f}GB")
            
            for gpu_id, info in memory_info.items():
                self.logger.info(f"GPU {gpu_id}: {info['utilization']:.1f}% 使用率 "
                               f"({info['reserved_gb']:.1f}GB/{info['total_gb']:.1f}GB)")
            self.logger.info("===================")
            self._status_logged = True
        else:
            # 後續顯示時，只顯示簡化信息
            if reason:
                self.logger.info(f"🔄 GPU狀態更新 ({reason}): 可用 {available_memory:.1f}GB/{total_memory:.1f}GB")
            else:
                self.logger.info(f"🔄 GPU記憶體: 可用 {available_memory:.1f}GB/{total_memory:.1f}GB")
        
        self._last_log_time = current_time


class OptimizedEmbeddingModel:
    """優化的嵌入模型包裝器，支持8bit量化和多GPU"""
    
    def __init__(self, model, tokenizer, device, max_length=512, batch_size=32, gpu_id=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gpu_id = gpu_id
        self.max_length = max_length
        self.batch_size = batch_size
        
    def encode(self, texts, batch_size=None, convert_to_tensor=True, device=None, **kwargs):
        """編碼文本為嵌入向量 - 支持多GPU"""
        if isinstance(texts, str):
            texts = [texts]
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # 如果指定了GPU ID，設置當前設備
        if self.gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
        
        all_embeddings = []
        
        # 分批處理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # 獲取嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 使用 [CLS] token 或平均池化
                if hasattr(outputs, 'last_hidden_state'):
                    # 平均池化
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                else:
                    # 使用 pooler_output 如果可用
                    embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0][:, 0]
                
                # 正規化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings)
        
        # 合併所有批次
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_tensor:
            return final_embeddings
        else:
            return final_embeddings.cpu().numpy()


class GPUResourceManager:
    """GPU資源管理器 - 統一管理GPU相關功能"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gpu_manager = MultiGPUManager()
        self.use_multi_gpu = False
        self.device = self._setup_device_strategy()
        
        # 記錄GPU使用情況（僅在初始化時顯示一次）
        if torch.cuda.is_available():
            self.gpu_manager.log_gpu_status(force=True, reason="初始化")
    
    def _setup_device_strategy(self) -> str:
        """設置設備策略"""
        config_device = self.config['models']['llm']['device']
        
        if config_device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                self.logger.info(f"✅ 多GPU模式啟用，檢測到 {gpu_count} 張GPU")
                self.use_multi_gpu = True
                return "cuda"  # 返回cuda，具體分配由device_map處理
            else:
                self.logger.info("✅ 單GPU模式啟用")
                self.use_multi_gpu = False
                return "cuda:0"
        elif config_device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA 不可用，切換到 CPU")
            self.use_multi_gpu = False
            return "cpu"
        else:
            self.use_multi_gpu = False
            return config_device
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """獲取即時GPU狀態"""
        if not torch.cuda.is_available():
            return {"error": "CUDA不可用"}
        
        return {
            "gpu_count": len(self.gpu_manager.available_gpus),
            "memory_info": self.gpu_manager.get_gpu_memory_info(),
            "total_memory_gb": self.gpu_manager.get_total_gpu_memory(),
            "available_memory_gb": self.gpu_manager.get_available_memory(),
            "multi_gpu_enabled": self.use_multi_gpu
        }
    
    def optimize_gpu_memory(self):
        """優化GPU記憶體使用"""
        if not torch.cuda.is_available():
            return
        
        self.logger.info("開始GPU記憶體優化...")
        
        # 清理未使用的記憶體
        self.gpu_manager.clear_gpu_memory()
        
        # 記錄優化結果
        memory_info = self.gpu_manager.get_gpu_memory_info()
        total_freed = 0
        
        for gpu_id, info in memory_info.items():
            freed = info['total_gb'] - info['reserved_gb']
            total_freed += freed
            self.logger.info(f"GPU {gpu_id}: 釋放 {freed:.1f}GB 記憶體")
        
        self.logger.info(f"✅ 記憶體優化完成，總共釋放 {total_freed:.1f}GB")
    
    def cleanup_gpu_resources(self):
        """清理GPU資源"""
        self.logger.info("開始清理GPU資源...")
        
        # 記錄清理前的記憶體使用情況（簡化顯示）
        if torch.cuda.is_available():
            self.logger.info("🧹 開始清理GPU記憶體...")
            # 不顯示詳細狀態，避免日誌冗餘
        
        # 多GPU環境下的深度清理
        if torch.cuda.is_available():
            if self.use_multi_gpu:
                # 清理所有GPU的記憶體
                self.gpu_manager.clear_gpu_memory(self.gpu_manager.available_gpus)
                
                # 重置CUDA環境
                for gpu_id in self.gpu_manager.available_gpus:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            else:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        # Python垃圾回收
        gc.collect()
        
        # 記錄清理後的記憶體使用情況（僅在必要時顯示）
        if torch.cuda.is_available():
            self.gpu_manager.log_gpu_status(reason="記憶體清理完成")
        
        success_msg = "✅ GPU資源已清理"
        if self.use_multi_gpu:
            success_msg += f" (已清理 {len(self.gpu_manager.available_gpus)} 張GPU)"
        self.logger.info(success_msg)
