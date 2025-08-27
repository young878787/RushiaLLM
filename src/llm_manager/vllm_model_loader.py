"""
vLLM 模型載入和初始化模組
負責載入和管理使用 vLLM 引擎的 Qwen-8B 模型和嵌入模型
提供與 ModelLoader 相同的接口，支援無縫切換
"""

import logging
import torch
import gc
import os
import asyncio
import uuid
import signal
import time
import random
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path

# vLLM imports - 使用 TYPE_CHECKING 處理類型註解
if TYPE_CHECKING:
    from vllm import AsyncLLMEngine, LLMEngine, EngineArgs, SamplingParams

try:
    from vllm import AsyncLLMEngine, LLMEngine, EngineArgs, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from .gpu_manager import OptimizedEmbeddingModel


class VLLMModelLoader:
    """vLLM 模型載入器 - 提供與 ModelLoader 相同的接口"""
    
    def __init__(self, config: dict, gpu_resource_manager):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gpu_resource_manager = gpu_resource_manager
        
        # 檢查 vLLM 可用性
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM 未安裝或不可用。請安裝 vLLM: pip install vllm")
        
        # 模型組件
        self.llm_engine: Optional['AsyncLLMEngine'] = None
        self.llm_tokenizer: Optional[AutoTokenizer] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # vLLM 特定配置
        self.vllm_config = config.get('models', {}).get('llm', {}).get('vllm', {})
        self.tensor_parallel_size = self._calculate_tensor_parallel_size()
        self.pipeline_parallel_size = self.vllm_config.get('pipeline_parallel_size', 1)
        
        # 性能優化組件（與 ModelLoader 保持一致）
        self.static_cache_enabled = False
        self.torch_compile_enabled = False
        self.optimization_applied = []
        
        self.logger.info("🚀 vLLM 模型載入器初始化完成")
        self.logger.info(f"   Tensor Parallel: {self.tensor_parallel_size}")
        self.logger.info(f"   Pipeline Parallel: {self.pipeline_parallel_size}")
    
    def __del__(self):
        """析構函數 - 確保資源被清理"""
        try:
            if hasattr(self, 'llm_engine') and self.llm_engine is not None:
                self.logger.warning("⚠️ VLLMModelLoader 被析構時發現未清理的資源，正在執行清理...")
                self.cleanup_models()
        except Exception:
            # 析構函數中不應該拋出異常
            pass
    
    def _calculate_tensor_parallel_size(self) -> int:
        """計算 Tensor Parallel 大小"""
        available_gpus = len(self.gpu_resource_manager.gpu_manager.available_gpus)
        
        # 從配置讀取，或自動計算
        config_tp_size = self.vllm_config.get('tensor_parallel_size', 'auto')
        
        if config_tp_size == 'auto':
            # 自動計算：優先使用所有可用 GPU
            if available_gpus >= 4:
                return 4  # 使用4張卡做 Tensor Parallel
            elif available_gpus >= 2:
                return 2  # 使用2張卡
            else:
                return 1  # 單卡
        else:
            # 使用配置指定的值
            return min(int(config_tp_size), available_gpus)
    
    async def load_llm_model(self) -> bool:
        """載入主要的 LLM 模型 - vLLM 版本"""
        self.logger.info("🚀 載入 Qwen-8B 主模型 (vLLM引擎)...")
        
        model_path = self.config['models']['llm']['model_path']
        
        # 🔥 處理相對路徑 - 從工作目錄的上級目錄開始
        if not os.path.isabs(model_path):
            # 從 scrpitsV2/LLM 目錄向上兩級到達 RushiaLLM 根目錄
            current_dir = Path(__file__).parent.parent.parent.parent.parent  # 到達 RushiaLLM 根目錄
            model_path = str(current_dir / model_path)
            self.logger.info(f"🔧 相對路徑轉換: {self.config['models']['llm']['model_path']} -> {model_path}")
        
        # 驗證路徑是否存在
        if not Path(model_path).exists():
            self.logger.error(f"❌ 模型路徑不存在: {model_path}")
            raise FileNotFoundError(f"模型路徑不存在: {model_path}")
        
        try:
            # 先載入 tokenizer (與原版保持一致)
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
                cache_dir=None
            )
            
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # 配置 vLLM EngineArgs
            engine_args = self._create_vllm_engine_args(model_path)
            
            # 🔧 修補 vLLM 0.10.1.1 兼容性問題
            # AsyncLLMEngine.from_engine_args 期望的屬性在新版 EngineArgs 中被移除
            if not hasattr(engine_args, 'enable_log_requests'):
                engine_args.enable_log_requests = False
            
            # 記錄 vLLM 配置
            self.logger.info("🔧 vLLM 引擎配置:")
            self.logger.info(f"   模型路徑: {model_path}")
            self.logger.info(f"   Tensor Parallel: {engine_args.tensor_parallel_size}")
            self.logger.info(f"   GPU 記憶體利用率: {engine_args.gpu_memory_utilization}")
            self.logger.info(f"   最大模型長度: {engine_args.max_model_len}")
            self.logger.info(f"   量化: {engine_args.quantization}")
            self.logger.info(f"   兼容性修補: enable_log_requests = {getattr(engine_args, 'enable_log_requests', 'N/A')}")
            
            # 創建 AsyncLLMEngine
            self.logger.info("⏳ 正在創建 vLLM 異步引擎...")
            self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.logger.info("✅ vLLM 引擎創建成功")
            
            # 記錄載入後的記憶體使用情況
            if torch.cuda.is_available():
                self.gpu_resource_manager.gpu_manager.log_gpu_status(reason="vLLM模型載入完成")
            
            success_msg = f"✅ Qwen-8B vLLM模型載入成功 (TP={self.tensor_parallel_size})"
            if self.tensor_parallel_size > 1:
                success_msg += f" (多GPU並行: {self.tensor_parallel_size}張卡)"
            self.logger.info(success_msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Qwen-8B vLLM模型載入失敗: {e}")
            # 清理可能的部分初始化資源
            await self._cleanup_partial_vllm_resources()
            raise e  # 直接拋出錯誤，不進行回退
    
    def _create_vllm_engine_args(self, model_path: str) -> 'EngineArgs':
        """創建 vLLM EngineArgs - 兼容 vLLM 0.10.1.1，支持8bit量化"""
        llm_config = self.config['models']['llm']
        vllm_config = self.vllm_config
        
        # 🔥 8bit 量化配置 - 映射到 vLLM 支持的 bitsandbytes
        quantization_mode = llm_config.get('quantization', '8bit')
        vllm_quantization = None
        
        if quantization_mode == '8bit':
            # 使用 bitsandbytes 進行 8bit 量化
            vllm_quantization = 'bitsandbytes'
            self.logger.info(f"🔧 啟用量化: {quantization_mode} -> vLLM {vllm_quantization}")
        elif quantization_mode == '4bit':
            # 使用 AWQ 作為4bit替代
            vllm_quantization = 'awq'
            self.logger.info(f"🔧 啟用量化: {quantization_mode} -> vLLM {vllm_quantization}")
        else:
            vllm_quantization = None
            self.logger.info("🔧 未啟用量化")
        
        # 🚀 生成動態隨機種子 - 確保每次啟動都有不同的隨機性
        import random
        dynamic_seed = int(time.time() * 1000000) % 2147483647 + random.randint(0, 10000)
        self.logger.info(f"🎲 使用動態隨機種子: {dynamic_seed}")
        
        # 基礎配置
        args = EngineArgs(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            trust_remote_code=True,
            
            # 🚀 關鍵修復：添加動態隨機種子，確保每次生成都有隨機性
            seed=dynamic_seed,
            
            # 記憶體配置 - 與 model_loader 8bit 模式保持一致
            gpu_memory_utilization=vllm_config.get('gpu_memory_utilization', 0.8),  # 降低記憶體使用
            swap_space=vllm_config.get('swap_space', 0.5),  # 2GB swap space
            
            # 模型配置
            max_model_len=vllm_config.get('max_model_len', llm_config.get('max_length', 2048)),  # 與配置一致
            max_num_seqs=vllm_config.get('max_num_seqs', 128),  # 降低並發數
            max_num_batched_tokens=vllm_config.get('max_num_batched_tokens', None),
            
            # 🔥 量化配置 - 支持8bit等效
            quantization=vllm_quantization,
            
            # 數據類型 - 與 model_loader 的 FP16 保持一致
            dtype=vllm_config.get('dtype', 'half'),  # 使用 half (FP16)
            
            # KV 緩存配置
            kv_cache_dtype=vllm_config.get('kv_cache_dtype', 'auto'),
            
            # 性能配置 - 只保留支援的參數
            disable_log_stats=vllm_config.get('disable_log_stats', False),
        )
        
        return args
    
    async def load_embedding_model(self) -> bool:
        """載入嵌入模型 - 與原版 model_loader 的 FP16 分配策略完全一致"""
        self.logger.info("🚀 載入 Qwen3-Embedding-0.6B 嵌入模型 (FP16原版)...")
        
        model_path = self.config['models']['embedding']['model_path']
        
        # 🔥 處理相對路徑 - 從工作目錄的上級目錄開始
        if not os.path.isabs(model_path):
            # 從 scrpitsV2/LLM 目錄向上兩級到達 RushiaLLM 根目錄
            current_dir = Path(__file__).parent.parent.parent.parent.parent  # 到達 RushiaLLM 根目錄
            model_path = str(current_dir / model_path)
            self.logger.info(f"🔧 相對路徑轉換: {self.config['models']['embedding']['model_path']} -> {model_path}")
        
        # 驗證路徑是否存在
        if not Path(model_path).exists():
            self.logger.error(f"❌ 嵌入模型路徑不存在: {model_path}")
            raise FileNotFoundError(f"嵌入模型路徑不存在: {model_path}")
        
        # 🔥 使用與原版 model_loader 完全相同的 GPU 分配邏輯
        embedding_config = self.gpu_resource_manager.gpu_manager.get_memory_allocation_config("embedding")
        embedding_device = embedding_config.get("device", self.gpu_resource_manager.device)
        embedding_gpu_id = embedding_config.get("target_gpu")
        
        if embedding_gpu_id is not None:
            embedding_device = f"cuda:{embedding_gpu_id}"
            self.logger.info(f"🎯 嵌入模型將使用GPU {embedding_gpu_id} (統一分配)")
        else:
            self.logger.info(f"🎯 嵌入模型使用設備: {embedding_device}")
        
        # 記錄嵌入模型分配信息
        self.gpu_resource_manager.gpu_manager.log_memory_allocation_info("embedding")
        
        try:
            # 🔥 優先嘗試使用 OptimizedEmbeddingModel (與原版一致)
            try:
                # 載入 tokenizer
                embedding_tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=None
                )
                
                # 🔥 FP16原版配置 - 移除量化，與原版完全一致
                from transformers import AutoModel
                
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if embedding_device != "cpu" else torch.float32,
                    "low_cpu_mem_usage": True,
                    "cache_dir": None,
                    "use_cache": True,
                }
                
                # CPU模式特殊配置
                if embedding_device == "cpu":
                    model_kwargs.update({
                        "torch_dtype": torch.float32,
                        "device_map": "cpu"
                    })
                
                # 如果是多GPU且指定了特定GPU，設置device_map
                if embedding_gpu_id is not None:
                    model_kwargs["device_map"] = {"": embedding_gpu_id}
                elif embedding_device != "cpu":
                    model_kwargs["device_map"] = {"": embedding_device}
                
                # 載入FP16原版嵌入模型
                embedding_model = AutoModel.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                
                if embedding_device == "cpu":
                    embedding_model = embedding_model.to("cpu")
                
                # 創建與原版相同的 OptimizedEmbeddingModel 包裝器
                self.embedding_model = OptimizedEmbeddingModel(
                    model=embedding_model,
                    tokenizer=embedding_tokenizer,
                    device=embedding_device,
                    gpu_id=embedding_gpu_id,
                    max_length=self.config['models']['embedding']['max_length'],
                    batch_size=self.config['models']['embedding']['batch_size']
                )
                
                success_msg = "✅ Qwen3-Embedding FP16模型載入成功 (OptimizedEmbeddingModel模式)"
                if embedding_gpu_id is not None:
                    success_msg += f" (GPU {embedding_gpu_id})"
                else:
                    success_msg += f" ({embedding_device})"
                self.logger.info(success_msg)
                
                return True
                
            except Exception as optimized_error:
                self.logger.warning(f"OptimizedEmbeddingModel 載入失敗: {optimized_error}")
                self.logger.info("🔄 回退到SentenceTransformer標準載入方法...")
                
                # 回退到 SentenceTransformer（與原版回退邏輯一致）
                fallback_device = embedding_device if embedding_device != "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
                
                self.embedding_model = SentenceTransformer(
                    model_path,
                    device=fallback_device,
                    cache_folder=None
                )
                
                success_msg = f"✅ Qwen3-Embedding 模型載入成功 (SentenceTransformer回退模式, {fallback_device})"
                if embedding_gpu_id is not None:
                    success_msg += f" (GPU {embedding_gpu_id})"
                self.logger.info(success_msg)
                
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Qwen3-Embedding 模型載入失敗: {e}")
            raise e  # 直接拋出錯誤，不進行回退
        
        finally:
            # 記錄載入後的記憶體使用情況（與原版一致）
            if torch.cuda.is_available():
                self.gpu_resource_manager.gpu_manager.log_gpu_status(reason="嵌入模型載入完成")
    
    async def generate_response_vllm(
        self, 
        prompt: str, 
        sampling_params: Optional['SamplingParams'] = None,
        request_id: Optional[str] = None
    ) -> str:
        """使用 vLLM 引擎生成回應"""
        if self.llm_engine is None:
            raise RuntimeError("vLLM 引擎未初始化")
        
        try:
            # 使用默認採樣參數（如果未提供）
            if sampling_params is None:
                sampling_params = self._get_default_sampling_params()
            
            # 生成唯一的請求 ID
            if request_id is None:
                import uuid
                request_id = str(uuid.uuid4())
            
            # 提交生成請求
            results_generator = self.llm_engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            # 等待生成完成
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output is None:
                raise RuntimeError("vLLM 生成結果為空")
            
            # 提取生成的文本
            generated_text = final_output.outputs[0].text
            return generated_text
            
        except Exception as e:
            self.logger.error(f"vLLM 生成回應失敗: {e}")
            raise e
    
    def _get_default_sampling_params(self) -> 'SamplingParams':
        """獲取默認的採樣參數 - vLLM 兼容版本，支援動態隨機種子"""
        llm_config = self.config['models']['llm']
        vtuber_config = self.config.get('vtuber', {}).get('response', {})
        
        # 🚀 每次生成使用不同的動態隨機種子，確保回應的多樣性
        import random
        generation_seed = int(time.time() * 1000000) % 2147483647 + random.randint(0, 1000)
        
        # 🔥 vLLM SamplingParams 只支援特定參數，移除不支援的 length_penalty
        # 以 config.yaml 配置為主，若未設定則使用預設值
        return SamplingParams(
            temperature=llm_config.get('temperature', self.config['models']['llm'].get('temperature', 0.75)),
            top_p=llm_config.get('top_p', self.config['models']['llm'].get('top_p', 0.8)),
            top_k=llm_config.get('top_k', self.config['models']['llm'].get('top_k', 40)),
            max_tokens=vtuber_config.get('max_tokens', self.config.get('vtuber', {}).get('response', {}).get('max_tokens', 150)),
            min_tokens=vtuber_config.get('min_tokens', self.config.get('vtuber', {}).get('response', {}).get('min_tokens', 25)),
            repetition_penalty=llm_config.get('repetition_penalty', self.config['models']['llm'].get('repetition_penalty', 1.15)),
            # 🎲 添加動態種子，每次生成都不同
            seed=generation_seed,
            # 注意：vLLM 不支援 length_penalty，已移除
            stop=None,  # 可以根據需要添加停止標記
            include_stop_str_in_output=False
        )
    
    def create_sampling_params(self, **kwargs) -> 'SamplingParams':
        """創建自定義採樣參數，默認使用動態隨機種子"""
        default_params = self._get_default_sampling_params()
        
        # 🚀 如果沒有明確指定種子，使用新的動態種子（即使已有默認種子）
        if 'seed' not in kwargs:
            import random
            kwargs['seed'] = int(time.time() * 1000000) % 2147483647 + random.randint(0, 1000)
            self.logger.debug(f"🎲 創建新的動態種子: {kwargs['seed']}")
        
        # 更新參數
        for key, value in kwargs.items():
            if hasattr(default_params, key):
                setattr(default_params, key, value)
        
        return default_params
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """獲取文本嵌入向量 - 與原版 OptimizedEmbeddingModel 接口兼容"""
        if self.embedding_model is None:
            raise RuntimeError("嵌入模型未載入")
        
        try:
            # 🔥 檢查是否是 OptimizedEmbeddingModel 包裝器
            if hasattr(self.embedding_model, 'encode_batch'):
                # 使用 OptimizedEmbeddingModel 的批次編碼（與原版一致）
                embeddings = self.embedding_model.encode_batch(
                    texts,
                    convert_to_tensor=True
                )
            else:
                # 使用 SentenceTransformer 的標準編碼（回退模式）
                embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=self.config['models']['embedding']['batch_size'],
                    convert_to_tensor=True,
                    device=self.gpu_resource_manager.device
                )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"生成嵌入向量時發生錯誤: {e}")
            raise e
    
    def get_model_input_device(self) -> Optional[str]:
        """獲取模型輸入設備 - vLLM 版本"""
        if self.tensor_parallel_size > 1:
            # 多GPU情況下，返回第一個GPU
            return f"cuda:{self.gpu_resource_manager.gpu_manager.available_gpus[0]}"
        else:
            return self.gpu_resource_manager.device
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息 - vLLM 版本，支持8bit量化和FP16嵌入"""
        try:
            # 獲取量化信息 - 使用與載入時相同的映射邏輯
            quantization_info = self.config['models']['llm'].get('quantization', '8bit')
            
            # 映射到 vLLM 支持的量化方法
            if quantization_info == '8bit':
                vllm_quantization = 'bitsandbytes'
            elif quantization_info == '4bit':
                vllm_quantization = 'awq'
            else:
                vllm_quantization = None
            
            # 基礎LLM信息 - 與原版 model_loader 風格一致
            llm_info = {
                "model_type": "Qwen-8B",
                "engine": "vLLM",
                "precision": "8bit" if quantization_info == '8bit' else quantization_info,
                "quantization": f"{quantization_info} (vLLM {vllm_quantization})" if vllm_quantization else quantization_info,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "device": self.gpu_resource_manager.device,
                "multi_gpu": self.tensor_parallel_size > 1,
                "memory_usage": "Managed by vLLM"
            }
            
            # 基礎嵌入模型信息 - 與原版風格一致
            embedding_device = "Unknown"
            embedding_type = "Unknown"
            
            if self.embedding_model:
                if hasattr(self.embedding_model, 'device'):
                    embedding_device = str(self.embedding_model.device)
                    embedding_type = "OptimizedEmbeddingModel (FP16)"
                else:
                    embedding_device = getattr(self.embedding_model, '_target_device', self.gpu_resource_manager.device)
                    embedding_type = "SentenceTransformer (FP16)"
            
            embedding_info = {
                "model_type": "Qwen3-Embedding-0.6B",
                "precision": "FP16",
                "quantization": "FP16原版（無量化）",
                "engine": embedding_type,
                "device": embedding_device,
                "memory_usage": "Unknown"
            }
            
            # GPU信息 - 與原版詳細程度一致
            gpu_info = {}
            if torch.cuda.is_available():
                try:
                    memory_info = self.gpu_resource_manager.gpu_manager.get_gpu_memory_info()
                    
                    gpu_info = {
                        "total_gpus": len(self.gpu_resource_manager.gpu_manager.available_gpus),
                        "available_gpus": self.gpu_resource_manager.gpu_manager.available_gpus,
                        "total_memory_gb": self.gpu_resource_manager.gpu_manager.get_total_gpu_memory(),
                        "available_memory_gb": self.gpu_resource_manager.gpu_manager.get_available_memory(),
                        "per_gpu_info": {}
                    }
                    
                    for gpu_id, info in memory_info.items():
                        gpu_name = self.gpu_resource_manager.gpu_manager.gpu_info[gpu_id]['name']
                        gpu_info["per_gpu_info"][f"gpu_{gpu_id}"] = {
                            "name": gpu_name,
                            "total_gb": info['total_gb'],
                            "allocated_gb": info['allocated_gb'],
                            "reserved_gb": info['reserved_gb'],
                            "free_gb": info['free_gb'],
                            "utilization_percent": info['utilization']
                        }
                except Exception as gpu_error:
                    self.logger.warning(f"獲取GPU信息時出錯: {gpu_error}")
                    gpu_info = {"error": "Failed to get GPU info"}
            
            return {
                "llm_model": llm_info,
                "embedding_model": embedding_info,
                "gpu_cluster": gpu_info,
                "performance_optimization": {
                    "engine": "vLLM",
                    "quantization": f"8bit -> vLLM {vllm_quantization}" if vllm_quantization else "8bit",
                    "tensor_parallel": f"{self.tensor_parallel_size}x GPU",
                    "memory_management": "vLLM Optimized",
                    "batch_processing": "Continuous Batching",
                    "kv_cache": "PagedAttention",
                    "embedding_precision": "FP16原版"
                }
            }
            
        except Exception as e:
            self.logger.error(f"獲取模型信息失敗: {e}")
            return {"error": str(e)}
    
    async def _cleanup_partial_vllm_resources(self):
        """清理部分初始化的 vLLM 資源"""
        try:
            if hasattr(self, 'llm_engine') and self.llm_engine is not None:
                # vLLM 引擎清理
                try:
                    # 停止後台任務
                    if hasattr(self.llm_engine, '_background_tasks'):
                        self.logger.info("🔧 停止 vLLM 後台任務...")
                        for task in self.llm_engine._background_tasks:
                            if not task.done():
                                task.cancel()
                    
                    # 清理引擎核心
                    if hasattr(self.llm_engine, 'engine'):
                        engine = self.llm_engine.engine
                        
                        # 清理模型執行器
                        if hasattr(engine, 'model_executor'):
                            self.logger.info("🔧 清理模型執行器...")
                            model_executor = engine.model_executor
                            
                            # 🔥 強制終止工作進程
                            if hasattr(model_executor, 'workers'):
                                for worker in model_executor.workers:
                                    try:
                                        if hasattr(worker, 'cleanup'):
                                            worker.cleanup()
                                        # 強制終止進程
                                        if hasattr(worker, 'pid'):
                                            try:
                                                os.kill(worker.pid, signal.SIGTERM)
                                                time.sleep(0.1)  # 給時間正常退出
                                                os.kill(worker.pid, signal.SIGKILL)  # 強制終止
                                            except (OSError, ProcessLookupError):
                                                pass  # 進程可能已經終止
                                    except Exception:
                                        pass
                            
                            # 清理驅動器工作進程
                            if hasattr(model_executor, 'driver_worker'):
                                self.logger.info("🔧 清理驅動器工作進程...")
                                driver_worker = model_executor.driver_worker
                                try:
                                    if hasattr(driver_worker, 'model_runner'):
                                        if hasattr(driver_worker.model_runner, 'model'):
                                            del driver_worker.model_runner.model
                                    if hasattr(driver_worker, 'cleanup'):
                                        driver_worker.cleanup()
                                    # 強制終止驅動器進程
                                    if hasattr(driver_worker, 'pid'):
                                        try:
                                            os.kill(driver_worker.pid, signal.SIGTERM)
                                            time.sleep(0.1)
                                            os.kill(driver_worker.pid, signal.SIGKILL)
                                        except (OSError, ProcessLookupError):
                                            pass
                                except Exception:
                                    pass
                            
                            del engine.model_executor
                        
                        # 清理調度器
                        if hasattr(engine, 'scheduler'):
                            self.logger.info("🔧 清理調度器...")
                            del engine.scheduler
                        
                        # 清理緩存引擎
                        if hasattr(engine, 'cache_engine'):
                            self.logger.info("🔧 清理緩存引擎...")
                            del engine.cache_engine
                    
                    # 最終清理引擎
                    del self.llm_engine
                    self.llm_engine = None
                    self.logger.info("✅ vLLM 引擎清理完成")
                    
                except Exception as cleanup_error:
                    self.logger.warning(f"清理 vLLM 引擎時出錯: {cleanup_error}")
            
            # 清理分散式資源
            try:
                self.logger.info("🔧 清理分散式資源...")
                destroy_model_parallel()
                self.logger.info("✅ 分散式資源清理完成")
            except Exception as cleanup_error:
                self.logger.warning(f"清理 model parallel 時出錯: {cleanup_error}")
            
            # 🔥 強制終止所有殘留的 vLLM 工作進程
            try:
                self.logger.info("🔧 清理殘留的 vLLM 工作進程...")
                import psutil
                current_pid = os.getpid()
                
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['pid'] != current_pid:
                            cmdline = ' '.join(proc.info['cmdline'] or [])
                            if 'vllm' in cmdline.lower() or 'VllmWorkerProcess' in cmdline:
                                self.logger.info(f"   終止殘留進程: {proc.info['pid']}")
                                proc.terminate()
                                proc.wait(timeout=1)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                        pass
            except ImportError:
                self.logger.warning("psutil 未安裝，無法清理殘留進程")
            except Exception as process_error:
                self.logger.warning(f"清理殘留進程時出錯: {process_error}")
            
            # 強制GPU記憶體清理
            if torch.cuda.is_available():
                self.logger.info("🔧 執行強制GPU記憶體清理...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 等待所有CUDA操作完成
                gc.collect()
                torch.cuda.empty_cache()  # 再次清理
                self.logger.info("✅ GPU記憶體清理完成")
            
        except Exception as e:
            self.logger.warning(f"清理部分初始化資源時出錯: {e}")
    
    def cleanup_models(self):
        """統一的模型資源清理方法 - 確保一次 Ctrl+C 就能完全退出"""
        self.logger.info("🧹 開始清理 vLLM 模型資源...")
        
        try:
            # 第一步：清理 vLLM 引擎
            if self.llm_engine is not None:
                try:
                    self.logger.info("🔧 清理 vLLM 引擎...")
                    
                    # 停止所有後台任務
                    if hasattr(self.llm_engine, '_background_tasks'):
                        try:
                            for task in self.llm_engine._background_tasks:
                                if not task.done():
                                    task.cancel()
                        except Exception:
                            pass
                    
                    # 深度清理引擎組件
                    if hasattr(self.llm_engine, 'engine'):
                        engine = self.llm_engine.engine
                        
                        # 清理模型執行器和工作進程
                        if hasattr(engine, 'model_executor'):
                            model_executor = engine.model_executor
                            
                            # 清理所有工作進程
                            if hasattr(model_executor, 'workers'):
                                for worker in model_executor.workers:
                                    try:
                                        if hasattr(worker, 'cleanup'):
                                            worker.cleanup()
                                    except Exception:
                                        pass
                            
                            # 清理驅動器工作進程
                            if hasattr(model_executor, 'driver_worker'):
                                driver_worker = model_executor.driver_worker
                                try:
                                    if hasattr(driver_worker, 'model_runner'):
                                        if hasattr(driver_worker.model_runner, 'model'):
                                            del driver_worker.model_runner.model
                                    if hasattr(driver_worker, 'cleanup'):
                                        driver_worker.cleanup()
                                except Exception:
                                    pass
                            
                            del engine.model_executor
                        
                        # 清理調度器和緩存
                        for attr in ['scheduler', 'cache_engine']:
                            if hasattr(engine, attr):
                                try:
                                    delattr(engine, attr)
                                except Exception:
                                    pass
                    
                    # 最終清理引擎
                    del self.llm_engine
                    self.llm_engine = None
                    self.logger.info("✅ vLLM 引擎已清理")
                    
                except Exception as e:
                    self.logger.warning(f"清理 vLLM 引擎時出錯: {e}")
            
            # 第二步：強制終止所有 vLLM 相關進程
            try:
                self.logger.info("🔧 強制終止 vLLM 工作進程...")
                import psutil
                import signal
                import time
                
                current_pid = os.getpid()
                killed_count = 0
                
                # 查找並終止所有 vLLM 相關進程
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['pid'] != current_pid:
                            cmdline = ' '.join(proc.info['cmdline'] or [])
                            process_name = proc.info['name'] or ''
                            
                            # 檢查是否是 vLLM 工作進程
                            is_vllm_process = (
                                'VllmWorkerProcess' in cmdline or
                                'vllm_worker' in process_name.lower() or
                                ('vllm' in cmdline.lower() and 'worker' in cmdline.lower()) or
                                ('multiproc_worker' in cmdline.lower() and 'vllm' in cmdline.lower())
                            )
                            
                            if is_vllm_process:
                                self.logger.info(f"   終止 vLLM 工作進程 PID {proc.info['pid']}")
                                
                                # 嘗試優雅退出
                                proc.terminate()
                                try:
                                    proc.wait(timeout=1)  # 等待1秒
                                except psutil.TimeoutExpired:
                                    # 強制終止
                                    proc.kill()
                                    try:
                                        proc.wait(timeout=1)
                                    except psutil.TimeoutExpired:
                                        pass
                                
                                killed_count += 1
                                
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                        pass
                
                if killed_count > 0:
                    self.logger.info(f"✅ 已終止 {killed_count} 個 vLLM 工作進程")
                    time.sleep(0.5)  # 給進程終止一點時間
                else:
                    self.logger.info("✅ 沒有發現 vLLM 工作進程")
                    
            except ImportError:
                self.logger.warning("⚠️ psutil 未安裝，無法強制終止工作進程")
                self.logger.warning("   建議安裝: pip install psutil")
            except Exception as process_error:
                self.logger.warning(f"終止工作進程時出錯: {process_error}")
            
            # 第三步：清理分散式資源
            try:
                self.logger.info("🔧 清理分散式資源...")
                destroy_model_parallel()
                self.logger.info("✅ Model Parallel 資源已清理")
            except Exception as e:
                self.logger.warning(f"清理 Model Parallel 時出錯: {e}")
            
            # 第四步：清理其他模型組件
            components = [
                ('llm_tokenizer', 'Tokenizer'),
                ('embedding_model', '嵌入模型')
            ]
            
            for attr_name, display_name in components:
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    try:
                        component = getattr(self, attr_name)
                        # 如果組件有清理方法，調用它
                        if hasattr(component, 'cleanup'):
                            component.cleanup()
                        del component
                        setattr(self, attr_name, None)
                        self.logger.info(f"✅ {display_name}已清理")
                    except Exception as e:
                        self.logger.warning(f"清理{display_name}時出錯: {e}")
            
            # 第五步：強制GPU記憶體清理
            if torch.cuda.is_available():
                try:
                    self.logger.info("🔧 執行GPU記憶體清理...")
                    
                    # 多次清理確保徹底
                    for i in range(3):
                        torch.cuda.empty_cache()
                        gc.collect()
                        if i < 2:
                            time.sleep(0.1)
                    
                    # 同步所有設備
                    torch.cuda.synchronize()
                    
                    # 逐個設備清理
                    for device_id in range(torch.cuda.device_count()):
                        try:
                            with torch.cuda.device(device_id):
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                        except Exception:
                            pass
                    
                    self.logger.info("✅ GPU記憶體已清理")
                    
                    # 記錄清理後狀態
                    if hasattr(self, 'gpu_resource_manager'):
                        try:
                            self.gpu_resource_manager.gpu_manager.log_gpu_status(reason="vLLM完全清理")
                        except Exception:
                            pass
                            
                except Exception as e:
                    self.logger.warning(f"GPU記憶體清理時出錯: {e}")
            
            self.logger.info("✅ vLLM 模型資源已完全清理")
            
        except Exception as e:
            self.logger.error(f"清理過程中發生錯誤: {e}")
            # 即使出錯也要設置資源為 None
            self.llm_engine = None
            self.llm_tokenizer = None
            self.embedding_model = None
    
    async def _cleanup_partial_vllm_resources(self):
        """清理部分初始化的 vLLM 資源 - 內部使用"""
        # 調用統一的清理方法
        self.cleanup_models()
    
    # ====================================================
    # 兼容性接口 - 與原版 ModelLoader 保持一致
    # ====================================================
    
    @property
    def llm_model(self):
        """兼容性屬性：返回 vLLM 引擎"""
        return self.llm_engine
    
    def apply_static_kv_cache_optimization(self) -> bool:
        """vLLM 自動處理 KV 緩存，返回 True 表示已優化"""
        self.static_cache_enabled = True
        self.optimization_applied.append("PagedAttention (vLLM)")
        self.logger.info("✅ vLLM PagedAttention 已啟用（自動 KV 緩存優化）")
        return True
    
    def apply_torch_compile_optimization(self) -> bool:
        """vLLM 引擎已優化，無需額外編譯"""
        self.torch_compile_enabled = True
        self.optimization_applied.append("vLLM Engine Optimization")
        self.logger.info("✅ vLLM 引擎優化已啟用（內建優化）")
        return True
    
    async def load_llm_model_optimized(self) -> bool:
        """載入優化的 vLLM 模型"""
        self.logger.info("🚀 載入高度優化的 vLLM 模型...")
        
        # vLLM 已經是高度優化的，直接載入
        success = await self.load_llm_model()
        if not success:
            raise RuntimeError("vLLM 模型載入失敗")
        
        # 標記優化已應用
        self.apply_static_kv_cache_optimization()
        self.apply_torch_compile_optimization()
        
        self.logger.info("✅ vLLM 模型優化完成")
        self.logger.info("📊 vLLM 內建優化:")
        self.logger.info("   PagedAttention: KV緩存優化")
        self.logger.info("   Continuous Batching: 批次處理優化")
        self.logger.info("   Tensor Parallel: 多GPU並行")
        self.logger.info("   Memory Efficiency: 記憶體優化")
        
        return True
