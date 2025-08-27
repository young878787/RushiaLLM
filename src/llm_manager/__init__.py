"""
LLM Manager 模組 - 重構版
將龐大的LLM管理器拆分成多個專責模組，提供更好的維護性和擴展性
支援 Transformers 和 vLLM 兩種載入模式
"""

# 主要接口導出
from .llm_manager import LLMManager, create_llm_manager, get_system_info, diagnose_system

# 核心組件導出（供高級用戶使用）
from .gpu_manager import GPUResourceManager, MultiGPUManager, OptimizedEmbeddingModel
from .model_loader import ModelLoader, create_model_loader, validate_loading_mode_config, get_available_loading_modes
from .response_generator import ResponseGenerator

# vLLM 組件（條件導出）
try:
    from .vllm_model_loader import VLLMModelLoader
    VLLM_AVAILABLE = True
except ImportError:
    VLLMModelLoader = None
    VLLM_AVAILABLE = False

# 版本信息
__version__ = "2.1.0"  # 升級版本號，支援 vLLM
__author__ = "RushiaLLM Team"

# 公開接口
__all__ = [
    # 主要接口
    "LLMManager",
    "create_llm_manager",
    "get_system_info", 
    "diagnose_system",
    
    # 核心組件（高級用戶）
    "GPUResourceManager",
    "MultiGPUManager", 
    "OptimizedEmbeddingModel",
    "ModelLoader",
    "ResponseGenerator",
    
    # 工廠函數和配置
    "create_model_loader",
    "validate_loading_mode_config",
    "get_available_loading_modes",
    
    # vLLM 組件（如果可用）
    "VLLMModelLoader" if VLLM_AVAILABLE else None,
    "VLLM_AVAILABLE",
]

# 移除 None 值
__all__ = [item for item in __all__ if item is not None]


def get_module_info():
    """獲取模組信息"""
    return {
        "name": "LLM Manager",
        "version": __version__,
        "description": "重構版LLM管理器 - 模組化架構",
        "components": {
            "gpu_manager": "GPU資源管理和多GPU支援",
            "model_loader": "模型載入和初始化",  
            "response_generator": "回應生成和處理",
            "llm_manager": "統一管理接口"
        },
        "features": [
            "多GPU智能分配",
            "4bit/8bit量化優化", 
            "動態記憶體管理",
            "語義向量增強",
            "情感理解系統",
            "角色人格一致性"
        ]
    }
