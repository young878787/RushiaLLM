"""
LLM 模型管理器
負責載入和管理 Qwen-8B 主模型和 Qwen3-Embedding 嵌入模型
"""

import asyncio
import logging
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, AutoModel
)
from sentence_transformers import SentenceTransformer
import gc
from .filter.filter import ResponseFilter
from .core import RushiaPersonalityCore


class OptimizedEmbeddingModel:
    """優化的嵌入模型包裝器，支持8bit量化"""
    
    def __init__(self, model, tokenizer, device, max_length=512, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
    def encode(self, texts, batch_size=None, convert_to_tensor=True, device=None, **kwargs):
        """編碼文本為嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        if batch_size is None:
            batch_size = self.batch_size
        
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


class LLMManager:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型組件
        self.llm_model: Optional[AutoModelForCausalLM] = None
        self.llm_tokenizer: Optional[AutoTokenizer] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # 回應過濾器
        self.response_filter = ResponseFilter(config)
        
        # 核心人格模組
        self.personality_core = RushiaPersonalityCore()
        
        # 🔥 新增：對話計數器（用於動態系統提示詞）
        self.conversation_count = 0
        
        # 設備配置
        self.device = config['models']['llm']['device']
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA 不可用，切換到 CPU")
            self.device = "cpu"
    
    async def initialize(self):
        """初始化所有模型"""
        await self._load_llm_model()
        await self._load_embedding_model()
        await self._setup_vtuber_personality()
    
    async def _load_llm_model(self):
        """載入主要的 LLM 模型 (Qwen-8B)"""
        self.logger.info("載入 Qwen-8B 主模型...")
        
        model_path = self.config['models']['llm']['model_path']
        
        # 4bit 量化配置 (固定使用4bit以獲得最佳記憶體效率)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ) if self.device == "cuda" else None
        
        try:
            # 載入 tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 設置 pad_token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # 載入模型
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.llm_model = self.llm_model.to(self.device)
            
            self.logger.info("✅ Qwen-8B 模型載入成功 (4bit量化)")
            
        except Exception as e:
            self.logger.error(f"❌ Qwen-8B 模型載入失敗: {e}")
            raise
    
    async def _load_embedding_model(self):
        """載入嵌入模型 (Qwen3-Embedding-0.6B) 使用8bit量化"""
        self.logger.info("載入 Qwen3-Embedding-0.6B 嵌入模型 (8bit量化)...")
        
        model_path = self.config['models']['embedding']['model_path']
        
        try:
            # 8bit 量化配置
            embedding_quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=False,
                llm_int8_threshold=6.0
            )
            
            # 先載入原始模型進行量化
            from transformers import AutoModel
            
            # 載入 tokenizer
            embedding_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 載入模型並應用8bit量化
            embedding_model = AutoModel.from_pretrained(
                model_path,
                quantization_config=embedding_quantization_config if self.device == "cuda" else None,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                embedding_model = embedding_model.to(self.device)
            
            # 創建自定義的嵌入模型包裝器
            self.embedding_model = OptimizedEmbeddingModel(
                model=embedding_model,
                tokenizer=embedding_tokenizer,
                device=self.device,
                max_length=self.config['models']['embedding']['max_length'],
                batch_size=self.config['models']['embedding']['batch_size']
            )
            
            self.logger.info("✅ Qwen3-Embedding 模型載入成功 (8bit量化)")
            
        except Exception as e:
            self.logger.error(f"❌ Qwen3-Embedding 模型載入失敗: {e}")
            # 回退到原始方法
            self.logger.info("回退到標準載入方法...")
            try:
                self.embedding_model = SentenceTransformer(
                    model_path,
                    device=self.device
                )
                self.logger.info("✅ Qwen3-Embedding 模型載入成功 (標準模式)")
            except Exception as fallback_error:
                self.logger.error(f"❌ 標準載入也失敗: {fallback_error}")
                raise
    
    async def _setup_vtuber_personality(self):
        """設置 VTuber 角色人格 - 完全依賴 core.json"""
        # 載入核心人格數據
        if not self.personality_core.load_core_personality():
            self.logger.error("❌ 無法載入核心人格數據，系統無法運行")
            raise RuntimeError("核心人格數據載入失敗，請檢查 rushia_wiki/core.json 文件")
        
        # 完全使用核心人格生成系統提示詞
        self.system_prompt = self.personality_core.generate_system_prompt()
        
        # 記錄載入的角色信息
        identity = self.personality_core.get_character_identity()
        personality = self.personality_core.get_personality_traits()
        
        # 設置過濾器的角色名稱
        character_name = identity['name'].get('zh', '露西婭')
        self.response_filter.set_character_name(character_name)
        
        self.logger.info("✅ VTuber 角色人格設置完成")
        self.logger.info(f"   角色名稱: {character_name}")
        self.logger.info(f"   性格特徵: {', '.join(personality['primary_traits'])}")
        self.logger.info(f"   當前情緒: {self.personality_core.current_mood}")
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        conversation_history: Optional[List[tuple]] = None,  # 🔥 新增：對話歷史
        stream: bool = False,
        rag_enabled: bool = True
    ) -> str:
        """生成回應 - 支持動態系統提示詞和對話歷史的語義分析增強版"""
        try:
            # 🔥 增加對話計數
            self.conversation_count += 1
            
            # 🔥 修復：傳遞對話歷史到情感分析
            if hasattr(self.personality_core, 'analyze_emotional_triggers_enhanced'):
                emotional_analysis = self.personality_core.analyze_emotional_triggers_enhanced(
                    prompt, conversation_history  # 🔥 傳遞對話歷史
                )
            else:
                emotional_analysis = self.personality_core.analyze_emotional_triggers(prompt)
            
            # 根據情緒分析調整心情
            if emotional_analysis['suggested_mood'] != self.personality_core.current_mood:
                self.personality_core.update_mood(emotional_analysis['suggested_mood'])
            
            # 🔥 新增：創建上下文提示（用於動態系統提示詞）
            context_hints = self._create_context_hints(emotional_analysis, conversation_history)
            
            # 🔥 使用動態系統提示詞生成（如果可用）
            if hasattr(self.personality_core, 'generate_dynamic_system_prompt'):
                self.system_prompt = self.personality_core.generate_dynamic_system_prompt(
                    conversation_count=self.conversation_count,
                    context_hints=context_hints
                )
                self.logger.info(f"📝 使用動態系統提示詞 (第 {self.conversation_count} 次對話)")
            elif hasattr(self.personality_core, 'generate_system_prompt_enhanced'):
                self.system_prompt = self.personality_core.generate_system_prompt_enhanced()
            else:
                self.system_prompt = self.personality_core.generate_system_prompt()
            
            # 獲取回應提示
            response_hints = self.personality_core.get_contextual_response_hints(emotional_analysis)
            
            # 語義向量增強的上下文檢索 (只在RAG啟用時)
            enhanced_context = context
            if rag_enabled and not context:
                enhanced_context = await self._get_semantic_vector_enhanced_context(prompt, emotional_analysis)
            
            # 構建完整的提示（包含對話歷史）
            full_prompt = self._build_enhanced_prompt_with_history(
                prompt, enhanced_context, conversation_history, emotional_analysis, response_hints
            )
            
            # 獲取動態生成參數
            generation_config = self._get_adaptive_generation_config(emotional_analysis)
            
            # Tokenize
            inputs = self.llm_tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config['models']['llm']['max_length'] - 512
            ).to(self.device)
            
            # 生成回應 - 🔥 在執行器中運行，避免阻塞
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_model_response,
                inputs,
                generation_config
            )
            
            # 應用過濾器處理回應
            filtered_response = self.response_filter.filter_response(response)
            
            # 驗證回應有效性
            if not self.response_filter.validate_response(filtered_response):
                self.logger.warning("生成的回應無效，使用備用回應")
                return "嗯嗯，我需要想想呢～"
            
            return filtered_response
            
        except Exception as e:
            self.logger.error(f"生成回應時發生錯誤: {e}")
            return "抱歉，我現在無法回應，請稍後再試。"
    
    def _generate_model_response(self, inputs, generation_config) -> str:
        """同步生成模型回應（在執行器中運行）"""
        try:
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 解碼回應
            response = self.llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"模型生成失敗: {e}")
            return "抱歉，生成過程中出現問題。"
    
    def _build_prompt(
        self, 
        user_input: str, 
        context: Optional[str] = None,
        emotional_analysis: Optional[Dict[str, Any]] = None,
        response_hints: Optional[Dict[str, Any]] = None
    ) -> str:
        """構建完整的提示"""
        prompt_parts = [self.system_prompt]
        
        # 添加RAG上下文
        if context:
            prompt_parts.append(f"\n<|context|>\n{context}\n<|end_context|>")
        
        # 添加情緒分析信息（如果有顯著觸發）
        if emotional_analysis and emotional_analysis.get('trigger_strength', 0) >= 2:
            emotion_info = f"檢測到情緒觸發：{emotional_analysis['emotional_category']}"
            if emotional_analysis['detected_triggers']:
                trigger_words = [t['word'] for t in emotional_analysis['detected_triggers']]
                emotion_info += f"，觸發詞：{', '.join(trigger_words)}"
            prompt_parts.append(f"\n<|emotion|>{emotion_info}<|end_emotion|>")
        
        # 添加回應風格提示
        if response_hints and response_hints.get('response_style') != 'normal':
            style_hint = ""
            if response_hints['response_style'] == 'intense':
                style_hint = "請以強烈的情緒回應，可能包含憤怒或激動的語氣"
            elif response_hints['response_style'] == 'moderate':
                style_hint = "請以適度的情緒回應，表現出相應的感情"
            
            if response_hints.get('special_phrases'):
                style_hint += f"，可以使用這些表達：{', '.join(response_hints['special_phrases'])}"
            
            if style_hint:
                prompt_parts.append(f"\n<|style|>{style_hint}<|end_style|>")
        
        prompt_parts.append(f"\n<|user|>{user_input}<|end|>")
        
        # 獲取角色名稱 - 完全從 core.json 獲取
        character_name = "AI助手"  # 默認回退名稱
        if hasattr(self.personality_core, 'core_data') and self.personality_core.core_data:
            character_name = self.personality_core.get_character_identity()['name'].get('zh', '露西婭')
        
        prompt_parts.append(f"\n<|assistant|>{character_name}：")
        
        return "\n".join(prompt_parts)
    
    async def _get_semantic_vector_enhanced_context(self, user_input: str, emotional_analysis: Dict[str, Any]) -> Optional[str]:
        """基於語義向量增強的智能檢索"""
        try:
            if not hasattr(self, '_rag_system_ref'):
                return None
            
            # 使用新的語義向量增強搜索
            search_results = await self._rag_system_ref.semantic_enhanced_search(
                user_input, 
                emotional_analysis, 
                top_k=8  # 增加檢索數量以獲得更多樣性
            )
            
            if not search_results:
                self.logger.info("🔍 語義向量搜索未找到結果，嘗試標準搜索")
                # 回退到標準搜索
                search_results = await self._rag_system_ref.search(user_input, top_k=5)
            
            if search_results:
                return self._build_diverse_context(search_results, emotional_analysis)
            
            return None
            
        except Exception as e:
            self.logger.error(f"語義向量增強檢索失敗: {e}")
            # 回退到標準檢索
            return await self._get_enhanced_context(user_input, emotional_analysis)

    async def _get_enhanced_context(self, user_input: str, emotional_analysis: Dict[str, Any]) -> Optional[str]:
        """根據情緒分析獲取增強版上下文（回退方案）"""
        try:
            
            # 根據情緒狀態調整檢索策略
            current_mood = self.personality_core.current_mood
            
            # 確定檢索類別優先級
            category_priority = self._determine_search_category(emotional_analysis, current_mood)
            
            # 使用增強版RAG系統檢索
            if hasattr(self, '_rag_system_ref'):
                # 直接調用search方法，保留搜索日誌
                search_results = await self._rag_system_ref.search(
                    user_input, 
                    top_k=5, 
                    category_filter=category_priority
                )
                
                if search_results:
                    # 手動構建上下文，不再顯示額外的LLM日誌
                    context_parts = []
                    for result in search_results:
                        content = result['content'].strip()
                        metadata = result['metadata']
                        
                        # 構建來源信息
                        source_info = []
                        if metadata.get('category'):
                            source_info.append(f"類別: {metadata['category']}")
                        if metadata.get('section_title'):
                            source_info.append(f"章節: {metadata['section_title']}")
                        if metadata.get('filename'):
                            source_info.append(f"文件: {metadata['filename']}")
                        
                        source_str = " | ".join(source_info) if source_info else "未知來源"
                        similarity = result['similarity']
                        
                        context_parts.append(f"[{source_str} | 相關度: {similarity:.3f}]\n{content}")
                    
                    return "\n\n---\n\n".join(context_parts)
                
                return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"獲取增強上下文失敗: {e}")
            return None
    
    def _determine_search_category(self, emotional_analysis: Dict[str, Any], current_mood: str) -> Optional[str]:
        """根據情緒分析確定搜索類別優先級"""
        
        # 檢查是否有情緒觸發
        if emotional_analysis.get('trigger_strength', 0) >= 2:
            emotional_category = emotional_analysis.get('emotional_category', 'neutral')
            
            if emotional_category == 'angry':
                # 憤怒情緒 - 優先搜索詞彙和身份相關
                return 'vocabulary'
            elif emotional_category == 'happy':
                # 開心情緒 - 優先搜索互動和內容相關
                return 'relationships'
        
        # 根據當前心情確定類別
        if current_mood == 'gaming':
            return 'content'  # 遊戲模式優先搜索內容相關
        elif current_mood == 'embarrassed':
            return 'core_identity'  # 敏感模式優先搜索身份相關
        elif current_mood == 'protective':
            return 'relationships'  # 保護模式優先搜索關係相關
        
        # 默認不限制類別
        return None
    
    def set_rag_system_reference(self, rag_system):
        """設置RAG系統引用（用於增強檢索）"""
        self._rag_system_ref = rag_system
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """獲取文本嵌入向量"""
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.config['models']['embedding']['batch_size'],
                convert_to_tensor=True,
                device=self.device
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"生成嵌入向量時發生錯誤: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        try:
            llm_info = {
                "model_type": "Qwen-8B",
                "quantization": "4bit",
                "device": self.device,
                "memory_usage": "Unknown"
            }
            
            embedding_info = {
                "model_type": "Qwen3-Embedding-0.6B", 
                "quantization": "8bit",
                "device": self.device,
                "memory_usage": "Unknown"
            }
            
            # 嘗試獲取記憶體使用情況
            if torch.cuda.is_available() and self.device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_cached = torch.cuda.memory_reserved(0)
                
                llm_info["gpu_total"] = f"{gpu_memory // (1024**3)}GB"
                llm_info["gpu_allocated"] = f"{gpu_allocated // (1024**3)}GB"
                llm_info["gpu_cached"] = f"{gpu_cached // (1024**3)}GB"
                
                embedding_info.update(llm_info)
            
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
            
            return {
                "llm_model": llm_info,
                "embedding_model": embedding_info,
                "personality_core": personality_info
            }
            
        except Exception as e:
            self.logger.error(f"獲取模型信息失敗: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """清理模型資源"""
        if self.llm_model:
            del self.llm_model
        if self.embedding_model:
            del self.embedding_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self.logger.info("模型資源已清理")
    
    def _determine_search_focus(self, intent_category: str) -> str:
        """確定檢索重點"""
        focus_mapping = {
            'companionship_request': 'companionship_responses',  # 檢索陪伴相關的回應
            'emotional_support': 'comfort_expressions',         # 檢索安慰表達
            'question_asking': 'helpful_responses',             # 檢索樂於助人的回應
            'casual_chat': 'casual_interactions',               # 檢索日常互動
            'intimate_expression': 'intimate_responses',        # 檢索親密回應
            'unknown': 'general_responses'                      # 檢索通用回應
        }
        return focus_mapping.get(intent_category, 'general_responses')

    def _get_emotional_search_priority(self, emotional_state: str) -> Dict[str, Any]:
        """根據情感狀態確定檢索優先級"""
        priority_map = {
            'very_strong': {
                'top_k': 8,
                'category': 'content',
                'keywords': ['強烈負面情緒', '溫柔安慰', '心疼關心']
            },
            'moderate': {
                'top_k': 6,
                'category': 'content', 
                'keywords': ['中等負面情緒', '溫柔回應', '關心理解']
            },
            'mild': {
                'top_k': 5,
                'category': 'content',
                'keywords': ['平靜情緒', '自然親切', '可愛風格']
            }
        }
        return priority_map.get(emotional_state, priority_map['mild'])

    def _get_intimacy_style_filter(self, intimacy_score: float) -> Dict[str, Any]:
        """根據親密度獲取風格過濾器"""
        if intimacy_score > 2.0:
            return {
                'style': 'very_intimate',
                'keywords': ['親愛的', '寶貝', '愛你', '戀人', '撒嬌', '甜蜜']
            }
        elif intimacy_score > 1.0:
            return {
                'style': 'warm_close',
                'keywords': ['親近', '溫暖', '黏人', '可愛', '關心']
            }
        else:
            return {
                'style': 'friendly',
                'keywords': ['友善', '親切', '禮貌', '自然']
            }

    def _build_enhanced_query(self, user_input: str, search_focus: str, emotional_state: str) -> str:
        """構建增強檢索查詢"""
        
        # 基於檢索重點添加關鍵詞
        focus_keywords = {
            'companionship_responses': '陪伴 一起 溫暖',
            'comfort_expressions': '安慰 溫柔 關心',
            'helpful_responses': '幫助 解答 樂意',
            'casual_interactions': '聊天 輕鬆 可愛',
            'intimate_responses': '親密 撒嬌 甜蜜',
            'general_responses': '回應 互動'
        }
        
        # 基於情感狀態添加修飾詞
        emotional_modifiers = {
            'very_strong': '特別溫柔 深度關懷',
            'moderate': '溫柔 關心',
            'mild': '親切 自然'
        }
        
        enhanced_query = f"{user_input} {focus_keywords.get(search_focus, '')} {emotional_modifiers.get(emotional_state, '')}"
        return enhanced_query.strip()

    def _filter_by_response_style(self, search_results: List, response_style_filter: Dict[str, Any]) -> List:
        """根據回應風格過濾檢索結果"""
        if not search_results:
            return []
        
        filtered_results = []
        style_keywords = response_style_filter.get('keywords', [])
        
        for result in search_results:
            content = result['content'].lower()
            
            # 計算風格匹配度
            style_match_count = sum(1 for keyword in style_keywords if keyword in content)
            
            # 添加風格匹配分數
            result['style_score'] = style_match_count / len(style_keywords) if style_keywords else 0.5
            
            # 只保留有一定匹配度的結果
            if result['style_score'] > 0.1 or result['similarity'] > 0.7:
                filtered_results.append(result)
        
        # 按風格匹配度和相似度排序
        filtered_results.sort(key=lambda x: (x['style_score'], x['similarity']), reverse=True)
        
        return filtered_results[:5]

    def _build_diverse_context(self, search_results: List, emotional_analysis: Dict) -> str:
        """構建多樣化上下文"""
        if not search_results:
            return ""
        
        context_parts = []
        
        # 統計結果來源
        file_sources = {}
        for result in search_results:
            filename = result['metadata'].get('filename', 'unknown')
            if filename not in file_sources:
                file_sources[filename] = []
            file_sources[filename].append(result)
        
        # 記錄多樣性信息
        self.logger.info(f"🎯 上下文來源多樣性: {len(file_sources)} 個文件")
        for filename, results in file_sources.items():
            self.logger.info(f"   📄 {filename}: {len(results)} 個片段")
        
        # 構建分類上下文
        for result in search_results[:6]:  # 最多6個結果
            content = result['content'].strip()
            metadata = result['metadata']
            
            # 構建來源信息
            source_info = []
            if metadata.get('category'):
                source_info.append(f"類別: {metadata['category']}")
            if metadata.get('section_title'):
                source_info.append(f"章節: {metadata['section_title']}")
            if metadata.get('filename'):
                source_info.append(f"文件: {metadata['filename']}")
            
            source_str = " | ".join(source_info) if source_info else "未知來源"
            similarity = result['similarity']
            
            # 標記是否為補充結果
            result_type = "補充" if result.get('supplementary', False) else "主要"
            
            context_parts.append(f"[{result_type}] [{source_str} | 相關度: {similarity:.3f}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_enhanced_prompt_with_history(
        self, 
        user_input: str, 
        context: Optional[str] = None,
        conversation_history: Optional[List[tuple]] = None,
        emotional_analysis: Optional[Dict[str, Any]] = None,
        response_hints: Optional[Dict[str, Any]] = None
    ) -> str:
        """構建包含對話歷史的增強版提示詞"""
        prompt_parts = [self.system_prompt]
        
        # 🔥 新增：添加對話歷史上下文
        if conversation_history and len(conversation_history) > 0:
            history_context = self._build_conversation_context(conversation_history)
            prompt_parts.append(f"\n<|conversation_history|>\n{history_context}\n<|end_history|>")
        
        # 添加語義分析上下文
        if context:
            prompt_parts.append(f"\n<|context|>\n{context}\n<|end_context|>")
        
        # 添加情感理解信息（使用人性化語言）
        if emotional_analysis and emotional_analysis.get('emotional_intensity') != 'mild':
            emotion_guidance = emotional_analysis.get('response_guidance', '')
            intimacy_guidance = emotional_analysis.get('intimacy_guidance', '')
            intent_guidance = emotional_analysis.get('intent_guidance', '')
            
            human_guidance = []
            if emotion_guidance:
                human_guidance.append(f"情感理解：{emotion_guidance}")
            if intimacy_guidance:
                human_guidance.append(f"互動方式：{intimacy_guidance}")
            if intent_guidance:
                human_guidance.append(f"回應重點：{intent_guidance}")
            
            if human_guidance:
                guidance_text = "、".join(human_guidance)
                prompt_parts.append(f"\n<|guidance|>{guidance_text}<|end_guidance|>")
        
        # 添加回應風格提示（人性化表達）
        if response_hints and response_hints.get('response_style') != 'normal':
            style_hint = ""
            if response_hints['response_style'] == 'intense':
                style_hint = "露西亞現在需要表現得特別溫柔體貼，用最關心的語氣回應"
            elif response_hints['response_style'] == 'moderate':
                style_hint = "露西亞要比平常更溫暖一些，表現出關心和理解"
            
            if response_hints.get('special_phrases'):
                style_hint += f"，可以使用這些表達方式：{', '.join(response_hints['special_phrases'])}"
            
            if style_hint:
                prompt_parts.append(f"\n<|style|>{style_hint}<|end_style|>")
        
        prompt_parts.append(f"\n<|user|>{user_input}<|end|>")
        
        # 獲取角色名稱 - 完全從 core.json 獲取
        character_name = "AI助手"  # 默認回退名稱
        if hasattr(self.personality_core, 'core_data') and self.personality_core.core_data:
            character_name = self.personality_core.get_character_identity()['name'].get('zh', '露西婭')
        
        prompt_parts.append(f"\n<|assistant|>{character_name}：")
        
        return "\n".join(prompt_parts)
    
    def _build_conversation_context(self, conversation_history: List[tuple]) -> str:
        """構建對話歷史上下文（保留最近7輪）"""
        if not conversation_history:
            return ""
        
        # 只使用最近的7輪對話
        recent_history = conversation_history[-7:]
        
        context_parts = []
        for i, (user_msg, bot_response) in enumerate(recent_history, 1):
            # 限制每條消息的長度，避免提示詞過長
            user_msg_short = user_msg[:100] + "..." if len(user_msg) > 100 else user_msg
            bot_response_short = bot_response[:150] + "..." if len(bot_response) > 150 else bot_response
            
            context_parts.append(f"第{i}輪對話:")
            context_parts.append(f"用戶: {user_msg_short}")
            context_parts.append(f"露西亞: {bot_response_short}")
            context_parts.append("")  # 空行分隔
        
        return "\n".join(context_parts).strip()
    
    def _get_adaptive_generation_config(self, emotional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """基於情感理解動態調整生成參數"""
        
        base_config = {
            "max_new_tokens": self.config['vtuber']['response']['max_tokens'],
            "min_new_tokens": self.config['vtuber']['response'].get('min_tokens', 10),
            "temperature": self.config['models']['llm']['temperature'],
            "top_p": self.config['models']['llm']['top_p'],
            "top_k": self.config['models']['llm']['top_k'],
            "repetition_penalty": self.config['models']['llm'].get('repetition_penalty', 1.15),
            "no_repeat_ngram_size": self.config['models']['llm'].get('no_repeat_ngram_size', 3),
            "length_penalty": self.config['models']['llm'].get('length_penalty', 1.0),
            "early_stopping": self.config['models']['llm'].get('early_stopping', True),
            "do_sample": True,
        }
        
        # 只有在tokenizer已初始化時才添加token相關配置
        if self.llm_tokenizer is not None:
            base_config["pad_token_id"] = self.llm_tokenizer.pad_token_id
            base_config["eos_token_id"] = self.llm_tokenizer.eos_token_id
        
        # 根據情感強度調整創造性
        emotional_intensity = emotional_analysis.get('emotional_intensity', 'mild')
        
        if emotional_intensity == 'very_strong':
            # 強烈情感需要更有創造性和表達力的回應
            base_config["temperature"] = min(0.8, base_config["temperature"] + 0.2)
            base_config["top_p"] = min(0.9, base_config["top_p"] + 0.1)
            base_config["max_new_tokens"] = min(200, base_config["max_new_tokens"] + 50)  # 150+50=200 ≈ 100字
            
        elif emotional_intensity == 'moderate':
            # 中等情感需要適度的表達力
            base_config["temperature"] = min(0.7, base_config["temperature"] + 0.1)
            base_config["max_new_tokens"] = min(180, base_config["max_new_tokens"] + 30)  # 150+30=180 ≈ 90字  
        
        # 根據親密度調整回應長度和細膩度
        intimacy_score = emotional_analysis.get('intimacy_score', 0.0)
        
        if intimacy_score > 2.0:
            # 高親密度需要更長更細膩的回應，但控制在100字內
            base_config["max_new_tokens"] = min(200, base_config["max_new_tokens"] + 30)  # 最多200 ≈ 100字
            base_config["min_new_tokens"] = max(25, base_config["min_new_tokens"] + 5)   
            base_config["repetition_penalty"] = max(1.05, base_config["repetition_penalty"] - 0.1)
        
        # 根據意圖類型調整準確性
        detected_intent = emotional_analysis.get('detected_intent', '')
        
        if 'question' in detected_intent or detected_intent == 'asking_info':
            # 問題類需要詳細但準確的回應，控制在100字內
            base_config["temperature"] = max(0.5, base_config["temperature"] - 0.1)  # 輕微降低創造性
            base_config["top_k"] = max(30, base_config["top_k"] - 10)                # 輕微減少候選詞
            
            # 增加問題類回應的token數量但不超過限制
            base_config["max_new_tokens"] = min(200, base_config["max_new_tokens"] + 30)  # 最多200 ≈ 100字
            base_config["min_new_tokens"] = max(30, base_config["min_new_tokens"] + 20)   # 15+20=35
            base_config["repetition_penalty"] = max(1.05, base_config["repetition_penalty"] - 0.1)  # 減少重複懲罰
            
        elif detected_intent == 'intimate_expression':
            # 親密表達需要更有情感的回應
            base_config["temperature"] = min(0.75, base_config["temperature"] + 0.15)
            base_config["repetition_penalty"] = max(1.1, base_config["repetition_penalty"] - 0.05)
        
        # 🔥 新增：基於回應期望的動態調整
        response_expectation = emotional_analysis.get('response_expectation', 'normal')
        
        if response_expectation == 'detailed':
            # 詳細回應：問題、安慰、複雜話題，但控制在100字內
            base_config["max_new_tokens"] = min(200, base_config["max_new_tokens"] + 40)  # 最多200 ≈ 100字
            base_config["min_new_tokens"] = max(30, base_config["min_new_tokens"] + 10)
            base_config["temperature"] = min(0.75, base_config["temperature"] + 0.05)  # 稍微增加表達力
            
        elif response_expectation == 'short':
            # 簡短回應：確認、簡單問候
            base_config["max_new_tokens"] = max(200, base_config["max_new_tokens"] - 50)
            base_config["min_new_tokens"] = max(10, base_config["min_new_tokens"] - 5)
        
        # normal 保持基礎配置不變
        
        return base_config
    
    def _create_context_hints(self, emotional_analysis: Dict[str, Any], conversation_history: Optional[List[tuple]] = None) -> Dict[str, Any]:
        """創建上下文提示（用於動態系統提示詞）"""
        context_hints = {}
        
        # 1. 情感狀態提示
        emotional_intensity = emotional_analysis.get('emotional_intensity', 'mild')
        if emotional_intensity == 'very_strong':
            context_hints['emotional_state'] = 'sad'  # 需要安慰
        elif emotional_intensity == 'moderate':
            context_hints['emotional_state'] = 'excited'  # 有情緒波動
        else:
            context_hints['emotional_state'] = 'calm'  # 平靜
        
        # 2. 對話主題類型推測
        detected_intent = emotional_analysis.get('detected_intent', 'unknown')
        if detected_intent in ['question_asking', 'asking_info']:
            context_hints['topic_type'] = 'casual'
        elif detected_intent == 'intimate_expression':
            context_hints['topic_type'] = 'emotional'
        elif detected_intent == 'companionship_request':
            context_hints['topic_type'] = 'daily_life'
        elif self.personality_core.current_mood == 'gaming':
            context_hints['topic_type'] = 'gaming'
        else:
            context_hints['topic_type'] = 'casual'
        
        # 3. 對話歷史分析
        if conversation_history and len(conversation_history) > 0:
            # 分析對話長度和頻率
            if len(conversation_history) > 5:
                context_hints['conversation_depth'] = 'deep'
            elif len(conversation_history) > 2:
                context_hints['conversation_depth'] = 'moderate'
            else:
                context_hints['conversation_depth'] = 'new'
            
            # 分析最近的交互模式
            recent_user_messages = [msg[0] for msg in conversation_history[-3:]]
            total_length = sum(len(msg) for msg in recent_user_messages)
            avg_length = total_length / len(recent_user_messages) if recent_user_messages else 0
            
            if avg_length > 50:
                context_hints['interaction_style'] = 'detailed'
            elif avg_length > 20:
                context_hints['interaction_style'] = 'normal'
            else:
                context_hints['interaction_style'] = 'brief'
        
        # 4. 時間上下文
        import datetime
        current_hour = datetime.datetime.now().hour
        
        if 6 <= current_hour <= 11:
            context_hints['time_period'] = 'morning'
        elif 12 <= current_hour <= 17:
            context_hints['time_period'] = 'afternoon'
        elif 18 <= current_hour <= 22:
            context_hints['time_period'] = 'evening'
        else:
            context_hints['time_period'] = 'night'
        
        return context_hints
    
    def reset_conversation_count(self):
        """重置對話計數器（用於新對話開始）"""
        self.conversation_count = 0
        self.logger.info("對話計數器已重置")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """獲取對話統計信息"""
        return {
            'conversation_count': self.conversation_count,
            'current_mood': self.personality_core.current_mood,
            'intimacy_level': getattr(self.personality_core, 'intimacy_level', 0.0),
            'semantic_enabled': getattr(self.personality_core, '_semantic_enabled', False)
        }