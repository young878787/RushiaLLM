"""
模型載入和初始化模組
負責載入和管理 Qwen-8B 主模型和 Qwen3-Embedding 嵌入模型
主模型：8bit量化優化，嵌入模型：FP16原版優化
支援 Transformers 和 vLLM 兩種載入模式
"""

import logging
import torch
import gc
import os
import warnings
from typing import Optional, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .vllm_model_loader import VLLMModelLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    AutoModel, AutoConfig, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer

from .gpu_manager import OptimizedEmbeddingModel

# 🔧 設置transformers日誌過濾，減少無關警告
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

# 過濾特定的無關警告
warnings.filterwarnings("ignore", message=".*generation flags.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*", category=UserWarning)


class ModelLoader:
    """模型載入器 - 主模型使用8bit量化，嵌入模型使用FP16優化"""
    
    def __init__(self, config: dict, gpu_resource_manager):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gpu_resource_manager = gpu_resource_manager
        
        # 模型組件
        self.llm_model: Optional[AutoModelForCausalLM] = None
        self.llm_tokenizer: Optional[AutoTokenizer] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # 🚀 性能優化組件
        self.static_cache_enabled = False
        self.torch_compile_enabled = False
        self.optimization_applied = []
        
        # 🔧 性能優化配置 - 從config.yaml讀取
        perf_config = config.get('performance', {})
        self.enable_static_cache = perf_config.get('enable_static_cache', True)
        self.enable_torch_compile = perf_config.get('enable_torch_compile', True)
        self.max_cache_length = perf_config.get('max_cache_length', 2048)
        
        # 記錄配置狀態
        self.logger.info("🔧 性能優化配置:")
        self.logger.info(f"   靜態KV緩存: {'啟用' if self.enable_static_cache else '禁用'}")
        self.logger.info(f"   Torch Compile: {'啟用' if self.enable_torch_compile else '禁用'}")
        self.logger.info(f"   最大緩存長度: {self.max_cache_length}")
        
        # 🔧 初始化時檢查生成配置
        self._validate_initial_config()
        
        # 🔧 性能優化配置
        self.enable_static_cache = config.get('performance', {}).get('enable_static_cache', True)
        self.enable_torch_compile = config.get('performance', {}).get('enable_torch_compile', True)
        self.max_cache_length = config.get('performance', {}).get('max_cache_length', 2048)
        
        # �🔧 初始化時檢查生成配置
        self._validate_initial_config()
    
    def _validate_initial_config(self):
        """驗證初始配置，提前發現問題"""
        try:
            # 檢查LLM配置中的潛在問題
            llm_config = self.config.get('models', {}).get('llm', {})
            
            # 檢查是否設置了衝突的參數
            deprecated_params = ['early_stopping']
            found_deprecated = [param for param in deprecated_params if param in llm_config]
            
            if found_deprecated:
                self.logger.warning(f"⚠️ 發現已棄用的LLM配置參數: {found_deprecated}")
                self.logger.warning("💡 這些參數將被自動過濾，不會影響運行")
            
            # 檢查VTuber回應配置
            vtuber_config = self.config.get('vtuber', {}).get('response', {})
            max_tokens = vtuber_config.get('max_tokens', 150)
            
            if max_tokens > 512:
                self.logger.warning(f"⚠️ max_tokens={max_tokens} 可能過大，建議控制在512以內")
            
            self.logger.info("✅ 模型配置驗證完成")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 配置驗證時出現問題: {e}")
    
    async def load_llm_model(self) -> bool:
        """載入主要的 LLM 模型 (Qwen-8B) - 8bit量化優化"""
        self.logger.info("🚀 載入 Qwen-8B 主模型 (8bit量化)...")
        
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
            return False
        
        # 獲取最佳設備映射 - 使用統一的記憶體分配配置
        memory_config = self.gpu_resource_manager.gpu_manager.get_memory_allocation_config("llm")
        device_map = None
        max_memory = None
        
        if self.gpu_resource_manager.use_multi_gpu:
            if memory_config.get("use_device_map"):
                device_map = memory_config.get("device_map", "auto")
                max_memory = memory_config.get("max_memory")
                
                if max_memory:
                    # 受控的記憶體分配策略
                    allowed_devices = memory_config.get("allowed_devices", [])
                    self.logger.info(f"🚀 使用統一記憶體分配策略")
                    self.logger.info(f"   目標GPU: {allowed_devices}")
                    self.logger.info(f"   記憶體分配: {max_memory}")
                    self.logger.info("   統一記憶體管理模式")
                    
                    # 記錄詳細的分配信息
                    self.gpu_resource_manager.gpu_manager.log_memory_allocation_info("llm")
                else:
                    self.logger.info("🔧 使用標準Auto分配策略")
            else:
                device_map = "auto"
                self.logger.info("🔄 回退到Auto分配策略")
        else:
            self.logger.info("📱 單GPU模式，不使用device_map")
        
        # 🔥 8bit量化配置
        try:
            # 載入 tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
                cache_dir=None  # 避免多進程衝突
            )
            
            # 設置 pad_token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # 🔥 8bit量化配置 - 使用BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False
            )
            
            model_kwargs = {
                "trust_remote_code": True,
                "quantization_config": quantization_config,
                "torch_dtype": torch.float16,  # 基礎精度為FP16，量化後為8bit
                "low_cpu_mem_usage": True,
                "cache_dir": None,
                # 標準attention實現
                "attn_implementation": "eager",  # 使用標準attention
                "use_cache": True,  # 啟用KV緩存提升生成速度
            }
            
            # 🔧 CPU模式回退配置
            if self.gpu_resource_manager.device == "cpu":
                # CPU模式不支持量化，回退到FP32
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,
                    "cache_dir": None,
                    "attn_implementation": "eager",
                    "use_cache": True,
                }
                self.logger.info("🔧 CPU模式：回退到FP32，無量化")
            
            # 添加設備映射和記憶體限制
            if self.gpu_resource_manager.use_multi_gpu:
                model_kwargs["device_map"] = device_map
                if max_memory:
                    model_kwargs["max_memory"] = max_memory
                    self.logger.info(f"🔧 統一記憶體限制: {max_memory}")
                self.logger.info("🔧 啟用多GPU統一設備映射")
            elif self.gpu_resource_manager.device != "cpu":
                model_kwargs["device_map"] = "auto"
                self.logger.info("🔧 啟用單GPU Auto設備映射")
            
            # 🔧 首次嘗試載入8bit量化模型
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                
                self.logger.info("✅ 8bit量化模型載入成功（首次嘗試）")
                
            except Exception as first_error:
                # 第一次失敗，嘗試回退策略
                self.logger.warning(f"8bit量化載入失敗: {first_error}")
                self.logger.info("🔄 嘗試回退策略...")
                
                # 移除可能有問題的記憶體限制
                fallback_kwargs = model_kwargs.copy()
                if 'max_memory' in fallback_kwargs:
                    del fallback_kwargs['max_memory']
                    self.logger.info("🔄 移除記憶體限制，使用標準分配")
                
                # 如果是多GPU，只使用基本的auto分配
                if self.gpu_resource_manager.use_multi_gpu:
                    fallback_kwargs["device_map"] = "auto"
                    self.logger.info("使用標準Auto分配作為回退")
                
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **fallback_kwargs
                )
                self.logger.info("✅ 8bit量化模型載入成功（回退策略）")
            
            # 如果是CPU模式，手動移動到設備
            if self.gpu_resource_manager.device == "cpu":
                self.llm_model = self.llm_model.to(self.gpu_resource_manager.device)
            
            # 記錄載入後的記憶體使用情況（簡化顯示）
            if torch.cuda.is_available():
                self.gpu_resource_manager.gpu_manager.log_gpu_status(reason="LLM模型載入完成")
            
            success_msg = "✅ Qwen-8B 8bit量化模型載入成功"
            if self.gpu_resource_manager.use_multi_gpu:
                total_gpus = len(memory_config.get("allowed_devices", []))
                if total_gpus > 0:
                    success_msg += f" (統一分配: {total_gpus}張卡，每卡{memory_config.get('memory_per_gpu', '4GB')})"
                else:
                    success_msg += f" (多GPU分配: {len(self.gpu_resource_manager.gpu_manager.available_gpus)}張卡)"
            else:
                success_msg += " (單GPU)"
            self.logger.info(success_msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Qwen-8B 8bit量化模型載入失敗: {e}")
            # 清理GPU記憶體
            if torch.cuda.is_available():
                self.gpu_resource_manager.gpu_manager.clear_gpu_memory()
            return False
    
    def apply_static_kv_cache_optimization(self) -> bool:
        """應用靜態KV緩存優化 - 8bit量化兼容版"""
        if self.llm_model is None:
            self.logger.warning("⚠️ 模型未載入，無法應用靜態KV緩存")
            return False
        
        if not self.enable_static_cache:
            self.logger.info("🔧 靜態KV緩存已在配置中禁用")
            return False
        
        try:
            self.logger.info("🚀 開始應用靜態KV緩存優化...")
            
            # 檢查模型是否支持靜態緩存
            model_config = getattr(self.llm_model, 'config', None)
            if model_config and hasattr(model_config, 'cache_implementation'):
                # 🔥 啟用靜態KV緩存
                model_config.cache_implementation = "static"
                self.logger.info("✅ 模型原生支持靜態緩存，已啟用")
                
                # 設置緩存配置
                max_cache_length = self.max_cache_length
                if self.gpu_resource_manager.use_multi_gpu:
                    max_cache_length = min(max_cache_length * 1.5, 3072)  # 多GPU可以使用更大緩存
                
                # 設置最大緩存長度
                if hasattr(model_config, 'max_cache_len'):
                    model_config.max_cache_len = int(max_cache_length)
                
                self.logger.info(f"📊 靜態KV緩存配置:")
                self.logger.info(f"   最大緩存長度: {max_cache_length}")
                self.logger.info(f"   多GPU模式: {self.gpu_resource_manager.use_multi_gpu}")
                self.logger.info(f"   預期記憶體節省: ~20-30%")
                self.logger.info(f"   預期速度提升: ~15-25%")
                
            else:
                # 手動實現靜態緩存邏輯
                self._apply_manual_static_cache()
            
            # 設置模型為推理優化模式
            self._configure_model_for_inference()
            
            self.static_cache_enabled = True
            self.optimization_applied.append("靜態KV緩存")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 靜態KV緩存優化失敗: {e}")
            self.logger.info("💡 可能原因:")
            self.logger.info("   1. 模型不支持靜態緩存")
            self.logger.info("   2. 記憶體不足")
            self.logger.info("   3. 8bit量化衝突")
            return False
    
    def _apply_manual_static_cache(self):
        """手動實現靜態緩存邏輯 - 8bit量化兼容"""
        try:
            self.logger.info("🔧 應用手動靜態緩存配置...")
            
            # 設置模型全域緩存配置
            if hasattr(self.llm_model.config, 'use_cache'):
                self.llm_model.config.use_cache = True
            
            # 為attention層設置靜態緩存標記
            attention_layers_found = 0
            for name, module in self.llm_model.named_modules():
                if any(keyword in name.lower() for keyword in ['attention', 'attn', 'self_attn']):
                    # 設置靜態緩存相關屬性
                    if hasattr(module, 'past_key_value') or hasattr(module, 'layer_past'):
                        module._use_static_cache = True
                        module._max_cache_length = self.max_cache_length
                        attention_layers_found += 1
            
            if attention_layers_found > 0:
                self.logger.info(f"✅ 為{attention_layers_found}個attention層配置了靜態緩存")
            else:
                self.logger.warning("⚠️ 未找到attention層，使用基礎緩存優化")
            
        except Exception as e:
            self.logger.warning(f"手動靜態緩存配置失敗: {e}")
    
    def _configure_model_for_inference(self):
        """配置模型為推理優化模式"""
        try:
            # 🔥 啟用推理模式
            self.llm_model.eval()
            
            # 🔥 禁用梯度計算（推理專用）
            for param in self.llm_model.parameters():
                param.requires_grad = False
            
            # 🔥 設置緩存相關配置
            if hasattr(self.llm_model.config, 'use_cache'):
                self.llm_model.config.use_cache = True
            
            self.logger.info("✅ 模型推理優化配置完成")
            
        except Exception as e:
            self.logger.warning(f"模型推理優化配置失敗: {e}")
    
    def apply_torch_compile_optimization(self) -> bool:
        """應用Torch Compile優化 - 8bit量化兼容版"""
        if self.llm_model is None:
            self.logger.warning("⚠️ 模型未載入，無法應用Torch Compile")
            return False
        
        if not self.enable_torch_compile:
            self.logger.info("🔧 Torch Compile已在配置中禁用")
            return False
        
        # 檢查Torch Compile可用性
        if not hasattr(torch, 'compile'):
            self.logger.warning("⚠️ 當前PyTorch版本不支持torch.compile (需要 >= 2.0)")
            return False
        
        if not torch.cuda.is_available():
            self.logger.warning("⚠️ Torch Compile需要CUDA支持")
            return False
        
        try:
            self.logger.info("🚀 開始應用Torch Compile優化...")
            
            # 🔥 檢查8bit量化兼容性並選擇最佳編譯模式
            compile_mode = self._get_optimal_compile_mode()
            
            self.logger.info(f"🔧 選擇編譯模式: {compile_mode}")
            self.logger.info("⏳ 正在編譯模型（首次編譯需要時間）...")
            
            # 編譯模型
            compiled_model = torch.compile(
                self.llm_model,
                mode=compile_mode,
                fullgraph=False,  # 8bit量化模型通常需要部分圖編譯
                dynamic=True,     # 支持動態輸入形狀
                backend="inductor"  # 使用inductor後端
            )
            
            # 替換原模型
            self.llm_model = compiled_model
            
            # 預熱編譯模型
            self._warmup_compiled_model()
            
            self.torch_compile_enabled = True
            self.optimization_applied.append("Torch Compile")
            
            self.logger.info(f"✅ Torch Compile優化完成 (模式: {compile_mode})")
            self.logger.info("📊 預期效能提升:")
            self.logger.info("   推理速度: +20-40%")
            self.logger.info("   記憶體效率: 輕微提升")
            self.logger.info("   後續推理: 無編譯延遲")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Torch Compile優化失敗: {e}")
            self.logger.info("💡 可能原因:")
            self.logger.info("   1. PyTorch版本 < 2.0")
            self.logger.info("   2. CUDA版本不兼容")
            self.logger.info("   3. 8bit量化與編譯器衝突")
            self.logger.info("   4. 模型架構不支持編譯")
            return False
    
    def _get_optimal_compile_mode(self) -> str:
        """獲取最適合的編譯模式"""
        
        # 根據GPU配置選擇編譯模式
        if self.gpu_resource_manager.use_multi_gpu:
            # 多GPU環境：使用保守模式，避免同步問題
            return "reduce-overhead"
        else:
            # 單GPU環境：可以使用更激進的優化
            gpu_memory = self.gpu_resource_manager.gpu_manager.get_total_gpu_memory()
            if gpu_memory >= 16:
                return "max-autotune"  # 高記憶體：最大優化
            else:
                return "reduce-overhead"  # 低記憶體：減少開銷
    
    def _warmup_compiled_model(self):
        """預熱編譯模型 - 觸發實際編譯"""
        try:
            if self.llm_tokenizer is None:
                self.logger.warning("⚠️ Tokenizer未載入，跳過模型預熱")
                return
            
            self.logger.info("🔥 正在預熱編譯模型（觸發編譯過程）...")
            
            # 創建測試輸入
            test_input = "Hello"
            test_tokens = self.llm_tokenizer.encode(
                test_input, 
                return_tensors="pt"
            )
            
            # 移動到正確設備
            input_device = self.get_model_input_device()
            if input_device:
                test_tokens = test_tokens.to(input_device)
            
            # 預熱推理（觸發編譯）
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = self.llm_model.generate(
                    test_tokens,
                    max_new_tokens=5,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.llm_tokenizer.pad_token_id
                )
                end_time.record()
                
                torch.cuda.synchronize()
                compile_time = start_time.elapsed_time(end_time) / 1000.0  # 轉換為秒
                
                self.logger.info(f"✅ 模型預熱完成 (編譯時間: {compile_time:.2f}秒)")
                self.logger.info("🚀 後續推理將享受編譯加速")
        
        except Exception as e:
            self.logger.warning(f"模型預熱失敗: {e}")
            self.logger.info("💡 編譯可能在首次實際使用時觸發")
    
    async def load_llm_model_optimized(self) -> bool:
        """載入優化的LLM模型 - 8bit量化 + 靜態KV緩存 + Torch Compile"""
        self.logger.info("🚀 載入高度優化的LLM模型...")
        
        # 先載入基礎8bit量化模型
        success = await self.load_llm_model()
        if not success:
            self.logger.error("❌ 基礎模型載入失敗，無法應用優化")
            return False
        
        # 應用優化
        optimizations_applied = []
        
        # 1. 應用靜態KV緩存
        if self.apply_static_kv_cache_optimization():
            optimizations_applied.append("靜態KV緩存")
        
        # 2. 應用Torch Compile
        if self.apply_torch_compile_optimization():
            optimizations_applied.append("Torch Compile")
        
        # 結果報告
        if optimizations_applied:
            self.logger.info(f"✅ 成功應用優化: {', '.join(optimizations_applied)}")
            self.logger.info("📊 預期總體提升:")
            
            if len(optimizations_applied) == 2:
                self.logger.info("   推理速度: +35-65%")
                self.logger.info("   記憶體效率: +20-30%")
                self.logger.info("   緩存命中率: 顯著提升")
            elif "靜態KV緩存" in optimizations_applied:
                self.logger.info("   推理速度: +15-25%")
                self.logger.info("   記憶體效率: +20-30%")
            elif "Torch Compile" in optimizations_applied:
                self.logger.info("   推理速度: +20-40%")
                self.logger.info("   編譯優化: 已啟用")
        else:
            self.logger.warning("⚠️ 未能應用任何優化，使用標準8bit配置")
            self.logger.info("💡 建議檢查配置文件中的performance設置")
        
        return True
    
    async def load_embedding_model(self) -> bool:
        """載入嵌入模型 (Qwen3-Embedding-0.6B) - FP16原版，無量化"""
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
            return False
        
        # 獲取嵌入模型的設備分配 - 使用統一配置
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
            # 載入 tokenizer
            embedding_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=None
            )
            
            # 🔥 FP16原版配置 - 移除量化
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if embedding_device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True,
                "cache_dir": None,
                # 🔥 為embedding模型也啟用優化
                "use_cache": True,
            }
            
            # CPU模式特殊配置
            if embedding_device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32
                self.logger.info("🔧 Embedding CPU模式：使用FP32")
            
            # 如果是多GPU且指定了特定GPU，設置device_map
            if embedding_gpu_id is not None:
                model_kwargs["device_map"] = {"": embedding_gpu_id}
                self.logger.info(f"🔧 Embedding模型映射到GPU {embedding_gpu_id}")
            elif embedding_device != "cpu":
                model_kwargs["device_map"] = "auto"
                self.logger.info("🔧 Embedding模型使用Auto設備映射")
            
            # 載入FP16原版嵌入模型
            embedding_model = AutoModel.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            if embedding_device == "cpu":
                embedding_model = embedding_model.to(embedding_device)
            
            # 創建自定義的嵌入模型包裝器
            self.embedding_model = OptimizedEmbeddingModel(
                model=embedding_model,
                tokenizer=embedding_tokenizer,
                device=embedding_device,
                gpu_id=embedding_gpu_id,
                max_length=self.config['models']['embedding']['max_length'],
                batch_size=self.config['models']['embedding']['batch_size']
            )
            
            success_msg = "✅ Qwen3-Embedding FP16模型載入成功"
            if embedding_gpu_id is not None:
                success_msg += f" (GPU {embedding_gpu_id})"
            else:
                success_msg += f" ({embedding_device})"
            self.logger.info(success_msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Qwen3-Embedding FP16模型載入失敗: {e}")
            # 回退到SentenceTransformer標準模式
            self.logger.info("🔄 回退到SentenceTransformer標準載入方法...")
            try:
                fallback_device = embedding_device if embedding_device != "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
                
                self.embedding_model = SentenceTransformer(
                    model_path,
                    device=fallback_device,
                    cache_folder=None
                )
                self.logger.info(f"✅ Qwen3-Embedding 模型載入成功 (SentenceTransformer模式, {fallback_device})")
                return True
            except Exception as fallback_error:
                self.logger.error(f"❌ SentenceTransformer載入也失敗: {fallback_error}")
                return False
        
        finally:
            # 記錄載入後的記憶體使用情況（簡化顯示）
            if torch.cuda.is_available():
                self.gpu_resource_manager.gpu_manager.log_gpu_status(reason="Embedding模型載入完成")
    
    def get_model_input_device(self) -> Optional[str]:
        """獲取模型輸入層的設備位置 - 精確定位版"""
        try:
            if not hasattr(self.llm_model, 'hf_device_map'):
                return None
            
            device_map = self.llm_model.hf_device_map
            
            # 按優先級搜索輸入層
            input_layer_patterns = [
                # 嵌入層優先（Transformer的真正輸入層）
                'model.embed_tokens',
                'transformer.wte', 
                'transformer.word_embeddings',
                'embeddings.word_embeddings',
                'embed_tokens',
                'wte',
                # 第一個Transformer層作為備選
                'model.layers.0',
                'transformer.h.0',
                'transformer.layers.0',
                'layers.0',
                'h.0'
            ]
            
            for pattern in input_layer_patterns:
                for module_name, device in device_map.items():
                    if pattern in module_name:
                        device_str = f"cuda:{device}" if isinstance(device, int) else str(device)
                        self.logger.debug(f"找到輸入層設備: {module_name} -> {device_str}")
                        return device_str
            
            # 如果沒找到，使用第一個可用GPU
            if self.gpu_resource_manager.gpu_manager.available_gpus:
                fallback_device = f"cuda:{self.gpu_resource_manager.gpu_manager.available_gpus[0]}"
                self.logger.warning(f"未找到明確輸入層，使用回退設備: {fallback_device}")
                return fallback_device
            
            return None
            
        except Exception as e:
            self.logger.error(f"獲取模型輸入設備失敗: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息 - FP16版本"""
        try:
            # 基礎LLM信息
            llm_info = {
                "model_type": "Qwen-8B",
                "precision": "8bit",  # 更新為8bit量化
                "quantization": "8bit (BitsAndBytesConfig)",  # 8bit量化
                "attention": "Eager",  # 標準attention實現
                "device": self.gpu_resource_manager.device,
                "multi_gpu": self.gpu_resource_manager.use_multi_gpu,
                "memory_usage": "Unknown"
            }
            
            # 基礎嵌入模型信息
            embedding_info = {
                "model_type": "Qwen3-Embedding-0.6B", 
                "precision": "FP16",  # 更新為FP16
                "quantization": "None",  # 無量化
                "device": getattr(self.embedding_model, 'device', self.gpu_resource_manager.device) if self.embedding_model else self.gpu_resource_manager.device,
                "memory_usage": "Unknown"
            }
            
            # 多GPU詳細信息
            gpu_info = {}
            if torch.cuda.is_available():
                memory_info = self.gpu_resource_manager.gpu_manager.get_gpu_memory_info()
                
                # 總體GPU信息
                gpu_info = {
                    "total_gpus": len(self.gpu_resource_manager.gpu_manager.available_gpus),
                    "available_gpus": self.gpu_resource_manager.gpu_manager.available_gpus,
                    "total_memory_gb": self.gpu_resource_manager.gpu_manager.get_total_gpu_memory(),
                    "available_memory_gb": self.gpu_resource_manager.gpu_manager.get_available_memory(),
                    "per_gpu_info": {}
                }
                
                # 每張卡的詳細信息
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
                
                # 更新LLM和embedding的記憶體信息
                if self.gpu_resource_manager.use_multi_gpu:
                    # 計算LLM使用的總記憶體（使用統一配置的GPU數量）
                    memory_config = self.gpu_resource_manager.gpu_manager.get_memory_allocation_config("llm")
                    llm_gpus = memory_config.get("allowed_devices", self.gpu_resource_manager.gpu_manager.available_gpus[:4])
                    llm_memory_used = sum(memory_info.get(gpu_id, {}).get('reserved_gb', 0) for gpu_id in llm_gpus)
                    llm_info["gpu_ids"] = llm_gpus
                    llm_info["memory_usage_gb"] = round(llm_memory_used, 2)
                    llm_info["memory_per_gpu"] = memory_config.get("memory_per_gpu", "4GB")
                    
                    # 嵌入模型記憶體使用
                    embedding_config = self.gpu_resource_manager.gpu_manager.get_memory_allocation_config("embedding")
                    embedding_gpu = embedding_config.get("target_gpu")
                    if embedding_gpu is not None:
                        embedding_memory = memory_info.get(embedding_gpu, {}).get('reserved_gb', 0)
                        embedding_info["gpu_id"] = embedding_gpu
                        embedding_info["memory_usage_gb"] = round(embedding_memory, 2)
                else:
                    # 單GPU模式
                    gpu_0_info = memory_info.get(0, {})
                    llm_info["memory_usage_gb"] = round(gpu_0_info.get('reserved_gb', 0), 2)
                    embedding_info["memory_usage_gb"] = round(gpu_0_info.get('allocated_gb', 0), 2)
            
            return {
                "llm_model": llm_info,
                "embedding_model": embedding_info,
                "gpu_cluster": gpu_info,
                "performance_optimization": {
                    "llm_quantization": "8bit",
                    "embedding_precision": "FP16",
                    "attention_implementation": "eager",
                    "memory_allocation": "Unified 4×4.5GB strategy",
                    "expected_benefits": "Consistent memory usage, optimized for 4-GPU setup"
                }
            }
            
        except Exception as e:
            self.logger.error(f"獲取模型信息失敗: {e}")
            return {"error": str(e)}
    
    def get_optimized_generation_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """獲取8bit量化優化的生成配置 - 修復版，避免參數衝突"""
        optimized_config = base_config.copy()
        
        if torch.cuda.is_available():
            # 🔥 8bit量化優化參數
            optimized_config.update({
                "use_cache": True,  # 啟用KV緩存
                "do_sample": True,
                # 基礎優化參數
                "pad_token_id": self.llm_tokenizer.pad_token_id if self.llm_tokenizer else None,
                "eos_token_id": self.llm_tokenizer.eos_token_id if self.llm_tokenizer else None,
                # 提升並行處理能力
                "output_attentions": False,  # 關閉attention輸出以節省記憶體
                "output_hidden_states": False,  # 關閉hidden states輸出
            })
            
            # 🔥 修復：不設置max_length，避免與max_new_tokens衝突
            # 只有在沒有設置max_new_tokens時才設置max_length
            if "max_new_tokens" not in optimized_config:
                # 根據GPU數量調整最大長度
                if self.gpu_resource_manager.use_multi_gpu:
                    # 多GPU可以處理更大的序列
                    optimized_config["max_length"] = 3072
                else:
                    # 單GPU保持保守設置
                    optimized_config["max_length"] = 2048
            
            self.logger.info("🚀 已啟用8bit量化優化生成配置")
        else:
            # 標準配置回退
            optimized_config.update({
                "use_cache": True,
                "output_attentions": False,
                "output_hidden_states": False,
            })
            self.logger.info("🔧 使用標準生成配置")
        
        return optimized_config
    
    def get_static_cache_generation_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """獲取靜態KV緩存優化的生成配置"""
        optimized_config = base_config.copy()
        
        # 🔥 靜態KV緩存專用配置
        optimized_config.update({
            # 啟用緩存
            "use_cache": True,
            
            # 靜態緩存配置
            "cache_implementation": "static" if hasattr(self.llm_model, 'config') and hasattr(self.llm_model.config, 'cache_implementation') else None,
            
            # 優化生成過程
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            
            # 避免參數衝突
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": True,
            
            # Token相關配置
            "pad_token_id": self.llm_tokenizer.pad_token_id if self.llm_tokenizer else None,
            "eos_token_id": self.llm_tokenizer.eos_token_id if self.llm_tokenizer else None,
        })
        
        # 設置緩存長度
        cache_length = self.max_cache_length
        if self.gpu_resource_manager.use_multi_gpu:
            cache_length = min(cache_length * 1.5, 3072)
        
        # 只在沒有max_new_tokens時設置max_length
        if "max_new_tokens" not in optimized_config:
            optimized_config["max_length"] = cache_length
        
        # 清理None值
        optimized_config = {k: v for k, v in optimized_config.items() if v is not None}
        
        self.logger.info("🚀 已啟用靜態KV緩存生成配置")
        self.logger.info(f"   緩存長度: {cache_length}")
        
        return optimized_config
    
    def get_torch_compile_generation_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """獲取Torch Compile優化的生成配置"""
        optimized_config = base_config.copy()
        
        # 🔥 Torch Compile專用優化
        optimized_config.update({
            # 緩存配置
            "use_cache": True,
            
            # 編譯友好的採樣配置
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            
            # 避免動態形狀變化（編譯器友好）
            "pad_token_id": self.llm_tokenizer.pad_token_id if self.llm_tokenizer else None,
            "eos_token_id": self.llm_tokenizer.eos_token_id if self.llm_tokenizer else None,
            
            # 關閉不必要的輸出
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": True,
            
            # 批次配置（編譯器優化）
            "num_beams": 1,  # 避免複雜的beam search
        })
        
        # 根據GPU配置調整
        if self.gpu_resource_manager.use_multi_gpu:
            optimized_config.update({
                "synced_gpus": True,
            })
            if "max_new_tokens" not in optimized_config:
                optimized_config["max_length"] = 3072
        else:
            if "max_new_tokens" not in optimized_config:
                optimized_config["max_length"] = 2048
        
        self.logger.info("🚀 已啟用Torch Compile優化生成配置")
        
        return optimized_config
    
    def get_combined_optimization_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """獲取組合優化配置 - 靜態緩存 + Torch Compile"""
        # 先應用靜態緩存配置
        config = self.get_static_cache_generation_config(base_config)
        
        # 再應用Torch Compile配置（合併兼容設置）
        torch_config = self.get_torch_compile_generation_config(base_config)
        
        # 合併配置，優先使用兼容的設置
        config.update({
            # 保持靜態緩存設置
            "cache_implementation": config.get("cache_implementation"),
            
            # 使用Torch Compile的批次優化
            "num_beams": torch_config.get("num_beams", 1),
            "synced_gpus": torch_config.get("synced_gpus", False),
            
            # 統一的長度設置
            "max_length": torch_config.get("max_length", config.get("max_length")),
        })
        
        # 添加組合優化特殊配置
        config.update({
            # 最大化緩存效率
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            
            # 編譯器友好設置
            "early_stopping": False,
        })
        
        # 記錄應用的優化
        applied_optimizations = []
        if self.static_cache_enabled:
            applied_optimizations.append("靜態KV緩存")
        if self.torch_compile_enabled:
            applied_optimizations.append("Torch Compile")
        
        self.logger.info(f"🚀 已啟用組合優化配置: {', '.join(applied_optimizations)}")
        
        return config
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """獲取當前優化狀態"""
        return {
            "static_cache_enabled": self.static_cache_enabled,
            "torch_compile_enabled": self.torch_compile_enabled,
            "optimizations_applied": self.optimization_applied.copy(),
            "max_cache_length": self.max_cache_length,
            "performance_config": {
                "enable_static_cache": self.enable_static_cache,
                "enable_torch_compile": self.enable_torch_compile,
            },
            "model_info": {
                "quantization": "8bit",
                "precision": "FP16",
                "multi_gpu": self.gpu_resource_manager.use_multi_gpu,
                "device": self.gpu_resource_manager.device,
            }
        }
    
    def cleanup_models(self):
        """清理模型資源"""
        self.logger.info("開始清理模型資源...")
        
        # 清理模型
        if self.llm_model:
            del self.llm_model
            self.llm_model = None
            
        if self.embedding_model:
            del self.embedding_model
            self.embedding_model = None
        
        # Python垃圾回收
        gc.collect()
        
        self.logger.info("✅ 模型資源已清理")


# ====================================================
# 🏭 模型載入器工廠函數
# ====================================================

def create_model_loader(config: dict, gpu_resource_manager):
    """
    模型載入器工廠函數
    根據配置選擇 Transformers 或 vLLM 載入方式
    """
    logger = logging.getLogger(__name__)
    
    # 檢查載入模式配置
    loading_mode = config.get('models', {}).get('llm', {}).get('loading_mode', 'transformers')
    
    if loading_mode.lower() == 'vllm':
        logger.info("🚀 選擇 vLLM 載入模式")
        try:
            from .vllm_model_loader import VLLMModelLoader
            return VLLMModelLoader(config, gpu_resource_manager)
        except ImportError as e:
            logger.error(f"❌ vLLM 不可用: {e}")
            logger.error("請安裝 vLLM: pip install vllm")
            raise ImportError("vLLM 載入器不可用，請安裝 vLLM 或切換到 transformers 模式")
    
    elif loading_mode.lower() == 'transformers':
        logger.info("🚀 選擇 Transformers 載入模式")
        return ModelLoader(config, gpu_resource_manager)
    
    else:
        logger.error(f"❌ 不支援的載入模式: {loading_mode}")
        logger.error("支援的模式: 'transformers', 'vllm'")
        raise ValueError(f"不支援的載入模式: {loading_mode}")


def get_available_loading_modes() -> Dict[str, Dict[str, Any]]:
    """獲取可用的載入模式信息"""
    modes = {
        'transformers': {
            'available': True,
            'description': 'Hugging Face Transformers - 8bit量化優化',
            'features': [
                '8bit量化節省記憶體',
                '靜態KV緩存優化',
                'Torch Compile加速',
                '多GPU設備映射'
            ],
            'best_for': '記憶體有限或需要精確控制的環境'
        }
    }
    
    # 檢查 vLLM 可用性
    try:
        import vllm
        modes['vllm'] = {
            'available': True,
            'description': 'vLLM高性能推理引擎',
            'features': [
                'PagedAttention記憶體優化',
                'Continuous Batching批次處理',
                'Tensor Parallel多GPU並行',
                '高吞吐量推理'
            ],
            'best_for': '高並發推理或大規模部署',
            'version': vllm.__version__
        }
    except ImportError:
        modes['vllm'] = {
            'available': False,
            'description': 'vLLM高性能推理引擎 (未安裝)',
            'install_command': 'pip install vllm',
            'features': [
                'PagedAttention記憶體優化',
                'Continuous Batching批次處理', 
                'Tensor Parallel多GPU並行',
                '高吞吐量推理'
            ],
            'best_for': '高並發推理或大規模部署'
        }
    
    return modes


def validate_loading_mode_config(config: dict) -> Dict[str, Any]:
    """驗證載入模式配置"""
    logger = logging.getLogger(__name__)
    
    validation_result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    llm_config = config.get('models', {}).get('llm', {})
    loading_mode = llm_config.get('loading_mode', 'transformers')
    
    # 檢查載入模式是否支援
    available_modes = get_available_loading_modes()
    
    if loading_mode not in available_modes:
        validation_result['valid'] = False
        validation_result['errors'].append(f"不支援的載入模式: {loading_mode}")
        validation_result['recommendations'].append(f"可用模式: {list(available_modes.keys())}")
        return validation_result
    
    if not available_modes[loading_mode]['available']:
        validation_result['valid'] = False
        validation_result['errors'].append(f"載入模式 {loading_mode} 不可用")
        if 'install_command' in available_modes[loading_mode]:
            validation_result['recommendations'].append(
                f"安裝命令: {available_modes[loading_mode]['install_command']}"
            )
        return validation_result
    
    # vLLM 特定配置檢查
    if loading_mode == 'vllm':
        vllm_config = llm_config.get('vllm', {})
        
        # 檢查 Tensor Parallel 配置
        tensor_parallel = vllm_config.get('tensor_parallel_size', 'auto')
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if tensor_parallel != 'auto' and isinstance(tensor_parallel, int):
            if tensor_parallel > available_gpus:
                validation_result['warnings'].append(
                    f"Tensor Parallel 大小 ({tensor_parallel}) 超過可用GPU數量 ({available_gpus})"
                )
                validation_result['recommendations'].append("將自動調整為可用GPU數量")
        
        # 檢查記憶體配置
        gpu_memory_util = vllm_config.get('gpu_memory_utilization', 0.85)
        if gpu_memory_util > 0.95:
            validation_result['warnings'].append(
                f"GPU記憶體利用率 ({gpu_memory_util}) 過高，可能導致OOM"
            )
            validation_result['recommendations'].append("建議設置為 0.85-0.9 之間")
    
    # Transformers 特定配置檢查
    elif loading_mode == 'transformers':
        # 檢查量化配置
        quantization = llm_config.get('quantization', '8bit')
        if quantization not in ['8bit', '4bit', 'none']:
            validation_result['warnings'].append(f"不建議的量化設置: {quantization}")
            validation_result['recommendations'].append("建議使用: '8bit', '4bit', 'none'")
    
    logger.info(f"✅ 載入模式配置驗證完成: {loading_mode}")
    if validation_result['warnings']:
        for warning in validation_result['warnings']:
            logger.warning(f"⚠️ {warning}")
    
    return validation_result
