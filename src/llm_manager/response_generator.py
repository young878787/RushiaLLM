"""
回應生成和處理模組
負責生成回應、處理提示詞、情感分析整合和對話歷史處理
"""

import asyncio
import logging
import torch
import datetime
from typing import Optional, List, Dict, Any, Tuple


class ResponseGenerator:
    """回應生成器 - 負責生成和處理所有回應邏輯"""
    
    def __init__(self, config: dict, model_loader, gpu_resource_manager, personality_core, response_filter):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 組件引用
        self.model_loader = model_loader
        self.gpu_resource_manager = gpu_resource_manager
        self.personality_core = personality_core
        self.response_filter = response_filter
        
        # RAG系統引用（將由外部設置）
        self._rag_system_ref = None
        
        # 對話計數器（用於動態系統提示詞）
        self.conversation_count = 0
    
    def set_rag_system_reference(self, rag_system):
        """設置RAG系統引用（用於增強檢索）"""
        self._rag_system_ref = rag_system
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        conversation_history: Optional[List[tuple]] = None,
        stream: bool = False,
        rag_enabled: bool = True
    ) -> str:
        """生成回應 - 支持動態系統提示詞和對話歷史的語義分析增強版"""
        try:
            # 增加對話計數
            self.conversation_count += 1
            
            # 情感分析 - 支持對話歷史
            if hasattr(self.personality_core, 'analyze_emotional_triggers_enhanced'):
                emotional_analysis = self.personality_core.analyze_emotional_triggers_enhanced(
                    prompt, conversation_history
                )
            else:
                emotional_analysis = self.personality_core.analyze_emotional_triggers(prompt)
            
            
            # 根據情緒分析調整心情
            if emotional_analysis['suggested_mood'] != self.personality_core.current_mood:
                self.personality_core.update_mood(emotional_analysis['suggested_mood'])
            
            # 創建上下文提示（用於動態系統提示詞）
            context_hints = self._create_context_hints(emotional_analysis, conversation_history)
            
            # 使用動態系統提示詞生成（如果可用）
            if hasattr(self.personality_core, 'generate_dynamic_system_prompt'):
                system_prompt = self.personality_core.generate_dynamic_system_prompt(
                    conversation_count=self.conversation_count,
                    context_hints=context_hints
                )
                self.logger.info(f"📝 使用動態系統提示詞 (第 {self.conversation_count} 次對話)")
            elif hasattr(self.personality_core, 'generate_system_prompt_enhanced'):
                system_prompt = self.personality_core.generate_system_prompt_enhanced()
            else:
                system_prompt = self.personality_core.generate_system_prompt()
            
            # 獲取回應提示
            response_hints = self.personality_core.get_contextual_response_hints(emotional_analysis)
            
            # 語義向量增強的上下文檢索 (只在RAG啟用時)
            enhanced_context = context
            if rag_enabled and not context:
                enhanced_context = await self._get_semantic_vector_enhanced_context(prompt, emotional_analysis)
            
            # 構建完整的提示（包含對話歷史）
            full_prompt = self._build_enhanced_prompt_with_history(
                system_prompt, prompt, enhanced_context, conversation_history, emotional_analysis, response_hints
            )
            
            # 獲取動態生成參數
            generation_config = self._get_adaptive_generation_config(emotional_analysis)
            
            # 檢查是否使用 vLLM
            loading_mode = self.config.get('models', {}).get('llm', {}).get('loading_mode', 'transformers')
            
            if loading_mode == 'vllm':
                # 使用 vLLM 生成
                response = await self._generate_vllm_response(full_prompt, generation_config)
            else:
                # 使用 Transformers 生成
                # Tokenize
                inputs = self.model_loader.llm_tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config['models']['llm']['max_length'] - 512
                ).to(self.gpu_resource_manager.device)
                
                # 生成回應 - 在執行器中運行，避免阻塞
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
    
    async def _generate_vllm_response(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """使用 vLLM 引擎生成回應"""
        try:
            # 檢查是否有 vLLM 專用的生成方法
            if hasattr(self.model_loader, 'generate_response_vllm'):
                # 創建 vLLM 採樣參數
                sampling_params = self.model_loader.create_sampling_params(
                    temperature=generation_config.get('temperature', 0.75),
                    top_p=generation_config.get('top_p', 0.8),
                    top_k=generation_config.get('top_k', 40),
                    max_tokens=generation_config.get('max_new_tokens', 150),
                    min_tokens=generation_config.get('min_new_tokens', 25),
                    repetition_penalty=generation_config.get('repetition_penalty', 1.15),
                    length_penalty=generation_config.get('length_penalty', 1.0)
                )
                
                # 使用 vLLM 生成
                response = await self.model_loader.generate_response_vllm(
                    prompt=prompt,
                    sampling_params=sampling_params
                )
                
                self.logger.debug(f"vLLM 生成完成，長度: {len(response)}")
                return response
            else:
                self.logger.error("vLLM 模型載入器缺少 generate_response_vllm 方法")
                return "嗯嗯，我需要想想呢～"
                
        except Exception as e:
            self.logger.error(f"vLLM 生成回應時發生錯誤: {e}")
            return "抱歉，我現在無法回應，請稍後再試。"
    
    def _generate_model_response(self, inputs, generation_config) -> str:
        """同步生成模型回應（在執行器中運行）- 多GPU優化版"""
        try:
            # 如果是多GPU環境，確保輸入在正確的設備上
            if self.gpu_resource_manager.use_multi_gpu and hasattr(self.model_loader.llm_model, 'hf_device_map'):
                # 更精準地定位輸入層設備
                input_device = self.model_loader.get_model_input_device()
                current_device = str(inputs['input_ids'].device)
                
                if input_device and current_device != input_device:
                    self.logger.debug(f"移動輸入張量：{current_device} → {input_device}")
                    try:
                        inputs = {k: v.to(input_device) for k, v in inputs.items() if hasattr(v, 'to')}
                    except Exception as e:
                        self.logger.error(f"輸入張量設備移動失敗: {e}")
                        # 回退到CPU再移動到目標設備
                        inputs = {k: v.cpu().to(input_device) for k, v in inputs.items() if hasattr(v, 'to')}
            
            with torch.no_grad():
                # 多GPU環境下的智能同步策略
                if self.gpu_resource_manager.use_multi_gpu:
                    # 只在必要時同步，避免不必要的延遲
                    if len(self.gpu_resource_manager.gpu_manager.available_gpus) > 1:
                        # 檢查是否需要同步：有多個設備且模型分佈在多GPU上
                        if hasattr(self.model_loader.llm_model, 'hf_device_map') and len(set(self.model_loader.llm_model.hf_device_map.values())) > 1:
                            torch.cuda.synchronize()  # 僅在真正需要時同步
                
                # 優化：使用設備特定的生成配置
                effective_config = generation_config.copy()
                
                # 在多GPU環境下調整批處理策略
                if self.gpu_resource_manager.use_multi_gpu:
                    # 減少不必要的計算開銷
                    effective_config['use_cache'] = True
                    effective_config['output_attentions'] = False
                    effective_config['output_hidden_states'] = False
                
                outputs = self.model_loader.llm_model.generate(
                    **inputs,
                    **effective_config
                )
                
                # 智能同步 - 只在有跨設備操作時同步
                if self.gpu_resource_manager.use_multi_gpu and hasattr(self.model_loader.llm_model, 'hf_device_map'):
                    if len(set(self.model_loader.llm_model.hf_device_map.values())) > 1:
                        torch.cuda.synchronize()
            
            # 解碼回應
            response = self.model_loader.llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"GPU記憶體不足: {e}")
            # 嘗試清理記憶體並重試
            self.gpu_resource_manager.gpu_manager.clear_gpu_memory()
            try:
                # 降低批次大小重試
                with torch.no_grad():
                    outputs = self.model_loader.llm_model.generate(
                        **inputs,
                        max_new_tokens=min(generation_config.get('max_new_tokens', 150), 100),
                        **{k: v for k, v in generation_config.items() if k != 'max_new_tokens'}
                    )
                
                response = self.model_loader.llm_tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                self.logger.warning("⚠️  記憶體不足，已降低生成長度")
                return response
            except Exception as retry_error:
                self.logger.error(f"重試生成失敗: {retry_error}")
                return "抱歉，記憶體不足，無法生成回應。"
        
        except Exception as e:
            self.logger.error(f"模型生成失敗: {e}")
            return "抱歉，生成過程中出現問題。"
    
    def _build_enhanced_prompt_with_history(
        self, 
        system_prompt: str,
        user_input: str, 
        context: Optional[str] = None,
        conversation_history: Optional[List[tuple]] = None,
        emotional_analysis: Optional[Dict[str, Any]] = None,
        response_hints: Optional[Dict[str, Any]] = None
    ) -> str:
        """構建包含對話歷史的增強版提示詞"""
        prompt_parts = [system_prompt]
        
        # 添加對話歷史上下文
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
    
    async def _get_semantic_vector_enhanced_context(self, user_input: str, emotional_analysis: Dict[str, Any]) -> Optional[str]:
        """基於語義向量增強的智能檢索"""
        try:
            if not self._rag_system_ref:
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
            if self._rag_system_ref:
                # 直接調用search方法，保留搜索日誌
                search_results = await self._rag_system_ref.search(
                    user_input, 
                    top_k=5, 
                    category_filter=category_priority
                )
                
                if search_results:
                    # 手動構建上下文
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
            content = result['content']
            style_score = 0
            
            # 計算風格匹配度
            for keyword in style_keywords:
                if keyword in content:
                    style_score += 1
            
            result['style_score'] = style_score / len(style_keywords) if style_keywords else 0
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
    
    def _get_adaptive_generation_config(self, emotional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """基於情感理解動態調整生成參數 - 修復版，移除無效參數"""
        
        base_config = {
            "max_new_tokens": self.config['vtuber']['response']['max_tokens'],
            "min_new_tokens": self.config['vtuber']['response'].get('min_tokens', 10),
            "temperature": self.config['models']['llm']['temperature'],
            "top_p": self.config['models']['llm']['top_p'],
            "top_k": self.config['models']['llm']['top_k'],
            "repetition_penalty": self.config['models']['llm'].get('repetition_penalty', 1.15),
            "no_repeat_ngram_size": self.config['models']['llm'].get('no_repeat_ngram_size', 3),
            "length_penalty": self.config['models']['llm'].get('length_penalty', 1.0),
            # 🔥 移除 early_stopping 參數 - 在新版transformers中無效
            "do_sample": True,
        }
        
        # 只有在tokenizer已初始化時才添加token相關配置
        if self.model_loader.llm_tokenizer is not None:
            base_config["pad_token_id"] = self.model_loader.llm_tokenizer.pad_token_id
            base_config["eos_token_id"] = self.model_loader.llm_tokenizer.eos_token_id
        
        # 🔥 應用FP16優化 - 但不設置max_length以避免衝突
        if hasattr(self.model_loader, 'get_optimized_generation_config'):
            base_config = self.model_loader.get_optimized_generation_config(base_config)
            self.logger.debug("✅ 已套用FP16生成優化")
        
        # 根據情感強度調整創造性
        emotional_intensity = emotional_analysis.get('emotional_intensity', 'mild')
        
        if emotional_intensity == 'very_strong':
            # 強烈情感需要更有創造性和表達力的回應
            base_config["temperature"] = min(0.8, base_config["temperature"] + 0.2)
            base_config["top_p"] = min(0.9, base_config["top_p"] + 0.1)
            base_config["max_new_tokens"] = min(200, base_config["max_new_tokens"] + 50)
            
        elif emotional_intensity == 'moderate':
            # 中等情感需要適度的表達力
            base_config["temperature"] = min(0.7, base_config["temperature"] + 0.1)
            base_config["max_new_tokens"] = min(180, base_config["max_new_tokens"] + 30)
        
        # 根據親密度調整回應長度和細膩度
        intimacy_score = emotional_analysis.get('intimacy_score', 0.0)
        
        if intimacy_score > 2.0:
            # 高親密度需要更長更細膩的回應，但控制在100字內
            base_config["max_new_tokens"] = min(200, base_config["max_new_tokens"] + 30)
            base_config["min_new_tokens"] = max(25, base_config["min_new_tokens"] + 5)
            base_config["repetition_penalty"] = max(1.05, base_config["repetition_penalty"] - 0.1)
        
        # 根據意圖類型調整準確性
        detected_intent = emotional_analysis.get('detected_intent', '')
        
        if 'question' in detected_intent or detected_intent == 'asking_info':
            # 問題類需要詳細但準確的回應
            base_config["temperature"] = max(0.5, base_config["temperature"] - 0.1)
            base_config["top_k"] = max(30, base_config["top_k"] - 10)
            base_config["max_new_tokens"] = min(200, base_config["max_new_tokens"] + 30)
            base_config["min_new_tokens"] = max(30, base_config["min_new_tokens"] + 20)
            base_config["repetition_penalty"] = max(1.05, base_config["repetition_penalty"] - 0.1)
            
        elif detected_intent == 'intimate_expression':
            # 親密表達需要更有情感的回應
            base_config["temperature"] = min(0.75, base_config["temperature"] + 0.15)
            base_config["repetition_penalty"] = max(1.1, base_config["repetition_penalty"] - 0.05)
        
        # 基於回應期望的動態調整
        response_expectation = emotional_analysis.get('response_expectation', 'normal')
        
        if response_expectation == 'detailed':
            # 詳細回應：問題、安慰、複雜話題，但控制在合理範圍內
            base_config["max_new_tokens"] = min(200, base_config["max_new_tokens"] + 40)
            base_config["min_new_tokens"] = max(30, base_config["min_new_tokens"] + 10)
            base_config["temperature"] = min(0.75, base_config["temperature"] + 0.05)
            
        elif response_expectation == 'short':
            # 簡短回應：確認、簡單問候
            base_config["max_new_tokens"] = max(50, base_config["max_new_tokens"] - 50)
            base_config["min_new_tokens"] = max(10, base_config["min_new_tokens"] - 5)
        
        # 🔥 新增：配置驗證和清理
        base_config = self._validate_generation_config(base_config)
        
        return base_config
    
    def _validate_generation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """驗證和清理生成配置，移除無效參數"""
        validated_config = config.copy()
        
        # 移除已棄用或無效的參數
        deprecated_params = [
            'early_stopping',  # 已在新版transformers中移除
            'forced_bos_token_id',  # 很少使用且可能造成問題
            'forced_eos_token_id',  # 很少使用且可能造成問題
        ]
        
        for param in deprecated_params:
            if param in validated_config:
                removed_value = validated_config.pop(param)
                self.logger.debug(f"🔧 移除無效參數: {param}={removed_value}")
        
        # 確保token ID的有效性
        for token_param in ['pad_token_id', 'eos_token_id', 'bos_token_id']:
            if token_param in validated_config and validated_config[token_param] is None:
                validated_config.pop(token_param, None)
                self.logger.debug(f"🔧 移除空的token參數: {token_param}")
        
        # 驗證數值參數的合理範圍
        if 'temperature' in validated_config:
            validated_config['temperature'] = max(0.1, min(2.0, validated_config['temperature']))
        
        if 'top_p' in validated_config:
            validated_config['top_p'] = max(0.1, min(1.0, validated_config['top_p']))
        
        if 'repetition_penalty' in validated_config:
            validated_config['repetition_penalty'] = max(1.0, min(2.0, validated_config['repetition_penalty']))
        
        # 確保max_new_tokens合理
        if 'max_new_tokens' in validated_config:
            validated_config['max_new_tokens'] = max(10, min(512, validated_config['max_new_tokens']))
        
        if 'min_new_tokens' in validated_config:
            validated_config['min_new_tokens'] = max(1, min(validated_config.get('max_new_tokens', 180), validated_config['min_new_tokens']))
        
        self.logger.debug(f"🔍 生成配置驗證完成: {len(validated_config)} 個參數")
        
        return validated_config
    
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
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """獲取對話統計信息"""
        return {
            'conversation_count': self.conversation_count,
            'current_mood': self.personality_core.current_mood,
            'intimacy_level': getattr(self.personality_core, 'intimacy_level', 0.0),
            'semantic_enabled': getattr(self.personality_core, '_semantic_enabled', False)
        }
    
    def reset_conversation_count(self):
        """重置對話計數器（用於新對話開始）"""
        self.conversation_count = 0
        self.logger.info("對話計數器已重置")
