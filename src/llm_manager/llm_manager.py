"""
LLM 模型管理器 - 重構版
負責統一管理所有LLM相關功能，提供一致的外部接口
"""

import asyncio
import logging
import torch
import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

# 導入重構後的模組
from .gpu_manager import GPUResourceManager
from .model_loader import create_model_loader, validate_loading_mode_config, get_available_loading_modes
from .response_generator import ResponseGenerator

# 導入依賴模組
import sys
from pathlib import Path
# 添加父目錄到路徑以支持絕對導入
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from filter.filter import ResponseFilter
from core import RushiaPersonalityCore


class LLMManager:
    """LLM管理器 - 重構版，統一管理所有LLM相關功能"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 🔥 驗證載入模式配置
        validation_result = validate_loading_mode_config(config)
        if not validation_result['valid']:
            for error in validation_result['errors']:
                self.logger.error(f"❌ 配置錯誤: {error}")
            for rec in validation_result['recommendations']:
                self.logger.info(f"💡 建議: {rec}")
            raise ValueError("載入模式配置無效")
        
        # 記錄警告和建議
        for warning in validation_result['warnings']:
            self.logger.warning(f"⚠️ {warning}")
        for rec in validation_result['recommendations']:
            self.logger.info(f"💡 建議: {rec}")
        
        # 🔥 核心組件初始化
        # GPU資源管理器
        self.gpu_resource_manager = GPUResourceManager(config)
        
        # 模型載入器 - 使用工廠函數創建
        self.model_loader = create_model_loader(config, self.gpu_resource_manager)
        
        # 記錄選擇的載入模式
        loading_mode = config.get('models', {}).get('llm', {}).get('loading_mode', 'transformers')
        self.logger.info(f"🚀 載入模式: {loading_mode}")
        
        # 核心人格模組
        self.personality_core = RushiaPersonalityCore()
        
        # 回應過濾器
        self.response_filter = ResponseFilter(config)
        
        # 回應生成器
        self.response_generator = ResponseGenerator(
            config, 
            self.model_loader, 
            self.gpu_resource_manager,
            self.personality_core, 
            self.response_filter
        )
        
        self.logger.info("✅ LLM管理器重構版初始化完成")
    
    async def initialize(self):
        """初始化所有模型和組件"""
        self.logger.info("🚀 開始初始化LLM管理器...")
        
        try:
            # 1. 載入LLM主模型
            if not await self.model_loader.load_llm_model():
                raise RuntimeError("LLM模型載入失敗")
            
            # 2. 載入嵌入模型
            if not await self.model_loader.load_embedding_model():
                raise RuntimeError("嵌入模型載入失敗")
            
            # 3. 設置VTuber人格
            if not await self._setup_vtuber_personality():
                raise RuntimeError("VTuber人格設置失敗")
            
            self.logger.info("✅ LLM管理器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ LLM管理器初始化失敗: {e}")
            return False
    
    async def _setup_vtuber_personality(self) -> bool:
        """設置 VTuber 角色人格 - 完全依賴 core.json"""
        try:
            # 載入核心人格數據
            if not self.personality_core.load_core_personality():
                self.logger.error("❌ 無法載入核心人格數據，系統無法運行")
                return False
            
            # 記錄載入的角色信息
            identity = self.personality_core.get_character_identity()
            personality = self.personality_core.get_personality_traits()
            
            # 設置過濾器的角色名稱
            character_name = identity['name'].get('zh', '露西婭')
            self.response_filter.set_character_name(character_name)
            
            # 🔥 修復：初始化語義分析系統
            if hasattr(self.personality_core, 'initialize_semantic_analysis'):
                semantic_success = self.personality_core.initialize_semantic_analysis()
                if semantic_success:
                    self.logger.info("✅ 語義分析系統已啟動")
                else:
                    self.logger.warning("⚠️ 語義分析系統初始化失敗，使用基礎模式")
            
            self.logger.info("✅ VTuber 角色人格設置完成")
            self.logger.info(f"   角色名稱: {character_name}")
            self.logger.info(f"   性格特徵: {', '.join(personality['primary_traits'])}")
            self.logger.info(f"   當前情緒: {self.personality_core.current_mood}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ VTuber人格設置失敗: {e}")
            return False
    
    # ==============================================
    # 🎯 主要公開接口
    # ==============================================
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        conversation_history: Optional[List[tuple]] = None,
        stream: bool = False,
        rag_enabled: bool = True
    ) -> str:
        """生成回應 - 主要公開接口"""
        return await self.response_generator.generate_response(
            prompt, context, conversation_history, stream, rag_enabled
        )
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """獲取文本嵌入向量"""
        try:
            embeddings = self.model_loader.embedding_model.encode(
                texts,
                batch_size=self.config['models']['embedding']['batch_size'],
                convert_to_tensor=True,
                device=self.gpu_resource_manager.device
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"生成嵌入向量時發生錯誤: {e}")
            raise
    
    def set_rag_system_reference(self, rag_system):
        """設置RAG系統引用（用於增強檢索）"""
        self.response_generator.set_rag_system_reference(rag_system)
    
    # ==============================================
    # 🔧 管理和監控接口
    # ==============================================
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取完整的模型信息"""
        try:
            # 基礎模型信息
            model_info = self.model_loader.get_model_info()
            
            # 添加載入模式信息
            loading_mode = self.config.get('models', {}).get('llm', {}).get('loading_mode', 'transformers')
            model_info["loading_mode"] = {
                "current_mode": loading_mode,
                "available_modes": get_available_loading_modes()
            }
            
            # 添加核心人格信息
            personality_info = {}
            if hasattr(self.personality_core, 'core_data') and self.personality_core.core_data:
                personality_info = {
                    "core_loaded": True,
                    "current_mood": self.personality_core.current_mood,
                    "character_name": self.personality_core.get_character_identity()['name'].get('zh', '露西婭'),
                    "personality_traits": len(self.personality_core.get_personality_traits()['primary_traits'])
                }
            else:
                personality_info = {"core_loaded": False}
            
            # 性能統計
            performance_info = {
                "conversation_count": self.response_generator.conversation_count,
                "multi_gpu_enabled": self.gpu_resource_manager.use_multi_gpu,
                "gpu_optimization": "enabled" if self.gpu_resource_manager.use_multi_gpu else "disabled",
                "optimizations_applied": getattr(self.model_loader, 'optimization_applied', [])
            }
            
            # 合併所有信息
            model_info["personality_core"] = personality_info
            model_info["performance"] = performance_info
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"獲取模型信息失敗: {e}")
            return {"error": str(e)}
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """獲取即時GPU狀態"""
        return self.gpu_resource_manager.get_gpu_status()
    
    def optimize_gpu_memory(self):
        """優化GPU記憶體使用"""
        self.gpu_resource_manager.optimize_gpu_memory()
    
    def diagnose_gpu_allocation(self) -> Dict[str, Any]:
        """診斷GPU分配狀況"""
        return self.gpu_resource_manager.gpu_manager.diagnose_gpu_allocation(
            self.model_loader.llm_model
        )
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """獲取對話統計信息"""
        return self.response_generator.get_conversation_stats()
    
    def reset_conversation_count(self):
        """重置對話計數器"""
        self.response_generator.reset_conversation_count()
    
    # ==============================================
    # 🧹 資源管理接口
    # ==============================================
    
    def cleanup(self):
        """清理所有資源"""
        self.logger.info("🧹 開始清理LLM管理器資源...")
        
        # 1. 清理模型資源
        self.model_loader.cleanup_models()
        
        # 2. 清理GPU資源
        self.gpu_resource_manager.cleanup_gpu_resources()
        
        self.logger.info("✅ LLM管理器資源清理完成")
    
    # ==============================================
    # 🔗 向後兼容接口
    # ==============================================
    
    @property
    def llm_model(self):
        """向後兼容：LLM模型訪問"""
        return self.model_loader.llm_model
    
    @property
    def llm_tokenizer(self):
        """向後兼容：LLM tokenizer訪問"""
        return self.model_loader.llm_tokenizer
    
    @property
    def embedding_model(self):
        """向後兼容：嵌入模型訪問"""
        return self.model_loader.embedding_model
    
    @property
    def device(self):
        """向後兼容：設備訪問"""
        return self.gpu_resource_manager.device
    
    @property
    def use_multi_gpu(self):
        """向後兼容：多GPU標識訪問"""
        return self.gpu_resource_manager.use_multi_gpu
    
    @property
    def conversation_count(self):
        """向後兼容：對話計數訪問"""
        return self.response_generator.conversation_count


# ==============================================
# 🏭 工廠函數
# ==============================================

async def create_llm_manager(config: dict) -> Optional[LLMManager]:
    """LLM管理器工廠函數 - 便捷創建和初始化"""
    try:
        # 創建管理器實例
        manager = LLMManager(config)
        
        # 初始化所有組件
        if await manager.initialize():
            return manager
        else:
            # 初始化失敗，清理資源
            manager.cleanup()
            return None
            
    except Exception as e:
        logging.error(f"❌ LLM管理器創建失敗: {e}")
        return None


# ==============================================
# 📊 系統信息和診斷工具
# ==============================================

def get_system_info() -> Dict[str, Any]:
    """獲取系統信息"""
    import platform
    import psutil
    
    return {
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        },
        "hardware": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "timestamp": str(datetime.datetime.now())
    }


def diagnose_system() -> Dict[str, Any]:
    """系統診斷工具"""
    diagnosis = {
        "system_info": get_system_info(),
        "issues": [],
        "recommendations": []
    }
    
    # 檢查CUDA可用性
    if not torch.cuda.is_available():
        diagnosis["issues"].append("CUDA不可用，只能使用CPU模式")
        diagnosis["recommendations"].append("安裝支持CUDA的PyTorch版本以獲得更好性能")
    
    # 檢查GPU記憶體
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            diagnosis["recommendations"].append("使用多GPU可以提高大模型載入速度和推理性能")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            if memory_gb < 8:
                diagnosis["issues"].append(f"GPU {i} 記憶體不足8GB，可能影響大模型載入")
    
    # 檢查系統記憶體
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 16:
        diagnosis["issues"].append("系統記憶體不足16GB，可能影響模型載入速度")
    
    return diagnosis
