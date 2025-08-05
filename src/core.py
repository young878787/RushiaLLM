"""
Rushia 核心人格模組
負責載入和管理 core.json 中的角色核心設定
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class RushiaPersonalityCore:
    """Rushia 核心人格管理器"""
    
    def __init__(self, core_json_path: str = "rushia_wiki/core.json"):
        self.logger = logging.getLogger(__name__)
        self.core_data = {}
        self.current_mood = "calm"
        self.core_json_path = core_json_path
        
        # 新增：語義分析能力
        self.semantic_manager = None
        self._semantic_enabled = True
        
        # 新增：情感記憶系統
        self.recent_emotions = []  # 最近的情感狀態
        self.intimacy_level = 0.0  # 當前親密度
        self.conversation_flow = "natural"  # 對話流暢度
        
    def load_core_personality(self) -> bool:
        """載入核心人格數據"""
        try:
            core_path = Path(self.core_json_path)
            if not core_path.exists():
                self.logger.error(f"核心人格文件不存在: {core_path}")
                return False
                
            with open(core_path, 'r', encoding='utf-8') as f:
                self.core_data = json.load(f)
                
            self.logger.info("✅ Rushia 核心人格數據載入成功")
            
            # 初始化語義分析系統
            self.initialize_semantic_analysis()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 載入核心人格數據失敗: {e}")
            return False
    
    def initialize_semantic_analysis(self):
        """初始化語義分析系統"""
        try:
            from .semantic_analysis import SemanticAnalysisManager
            self.semantic_manager = SemanticAnalysisManager(chat_instance=self)
            self.logger.info("✅ 情感理解系統已啟動")
            return True
        except ImportError as e:
            self.logger.warning(f"⚠️ 語義分析模組不可用，使用基礎模式: {e}")
            self._semantic_enabled = False
            return False
        except Exception as e:
            self.logger.error(f"❌ 語義分析系統初始化失敗: {e}")
            self._semantic_enabled = False
            return False
    
    def get_character_identity(self) -> Dict[str, Any]:
        """獲取角色身份信息"""
        character_core = self.core_data.get('character_core', {})
        return {
            'name': character_core.get('name', {}),
            'basic_info': character_core.get('basic_info', {}),
            'role': character_core.get('basic_info', {}).get('role', '死靈法師公主')
        }
    
    def get_personality_traits(self) -> Dict[str, Any]:
        """獲取性格特徵"""
        character_core = self.core_data.get('character_core', {})
        core_personality = character_core.get('core_personality', {})
        
        return {
            'primary_traits': core_personality.get('primary_traits', []),
            'voice_style': core_personality.get('voice_style', ''),
            'emotional_triggers': core_personality.get('emotional_triggers', {})
        }
    
    def get_signature_elements(self) -> Dict[str, Any]:
        """獲取標誌性元素"""
        character_core = self.core_data.get('character_core', {})
        signature = character_core.get('signature_elements', {})
        
        return {
            'greetings': signature.get('greetings', {}),
            'fan_name': signature.get('fan_name', 'ふぁんでっど'),
            'nicknames': signature.get('nicknames', []),
            'catchphrases': signature.get('catchphrases', [])
        }
    
    def get_content_style(self) -> Dict[str, Any]:
        """獲取內容風格"""
        character_core = self.core_data.get('character_core', {})
        content_style = character_core.get('content_style', {})
        
        return {
            'stream_types': content_style.get('stream_types', []),
            'gaming_personality': content_style.get('gaming_personality', ''),
            'interaction_style': content_style.get('interaction_style', '')
        }
    
    def get_emotional_system(self) -> Dict[str, Any]:
        """獲取情緒系統"""
        emotional_system = self.core_data.get('emotional_system', {})
        
        return {
            'mood_states': emotional_system.get('mood_states', {}),
            'trigger_words': emotional_system.get('trigger_words', {})
        }
    
    def analyze_emotional_triggers(self, text: str) -> Dict[str, Any]:
        """分析文本中的情緒觸發詞"""
        emotional_system = self.get_emotional_system()
        trigger_words = emotional_system.get('trigger_words', {})
        
        result = {
            'detected_triggers': [],
            'emotional_category': 'neutral',
            'trigger_strength': 0,
            'suggested_mood': self.current_mood
        }
        
        text_lower = text.lower()
        
        # 🔥 新增：情感關鍵詞擴展檢測
        emotion_keywords = {
            'positive_strong': ['愛', '喜歡', '開心', '高興', '快樂', '棒', '讚', '厲害', '可愛', '超棒', '太好了'],
            'positive_mild': ['好', '不錯', '還行', '謝謝', '感謝', 'nice', '讚'],
            'negative_strong': ['討厭', '生氣', '難過', '傷心', '痛苦', '煩躁', '氣死了', '超難過'],
            'negative_mild': ['累', '疲倦', '無聊', '普通', '還好'],
            'intimate': ['親愛的', '寶貝', '想你', '喜歡你', '愛你', '抱抱', '親親'],
            'questioning': ['為什麼', '怎麼', '什麼', '哪裡', '誰'],
            'emotional_support': ['安慰', '陪伴', '聊天', '傾聽', '理解', '孤單', '寂寞']
        }
        
        # 檢測情感關鍵詞
        for category, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text or keyword in text_lower:
                    strength = 3 if 'strong' in category else (2 if 'mild' in category else 1)
                    
                    result['detected_triggers'].append({
                        'word': keyword,
                        'category': category,
                        'strength': strength
                    })
                    
                    result['trigger_strength'] = max(result['trigger_strength'], strength)
                    
                    # 設置情感類別
                    if 'positive' in category:
                        result['emotional_category'] = 'happy'
                    elif 'negative' in category:
                        result['emotional_category'] = 'sad'
                    elif category == 'intimate':
                        result['emotional_category'] = 'intimate'
                        result['trigger_strength'] = max(result['trigger_strength'], 2)
        
        # 檢測負面觸發詞（最高優先級）
        negative_words = trigger_words.get('negative', [])
        for word in negative_words:
            if word in text or word.lower() in text_lower:
                result['detected_triggers'].append({
                    'word': word,
                    'category': 'negative',
                    'strength': 3  # 高強度
                })
                result['emotional_category'] = 'angry'
                result['trigger_strength'] = max(result['trigger_strength'], 3)
                result['suggested_mood'] = 'embarrassed'  # 身高敏感模式
        
        # 檢測正面觸發詞
        positive_words = trigger_words.get('positive', [])
        for word in positive_words:
            if word in text or word.lower() in text_lower:
                result['detected_triggers'].append({
                    'word': word,
                    'category': 'positive',
                    'strength': 2  # 中強度
                })
                if result['emotional_category'] == 'neutral':
                    result['emotional_category'] = 'happy'
                    result['trigger_strength'] = max(result['trigger_strength'], 2)
                    result['suggested_mood'] = 'calm'
        
        # 檢測中性觸發詞
        neutral_words = trigger_words.get('neutral', [])
        for word in neutral_words:
            if word in text or word.lower() in text_lower:
                result['detected_triggers'].append({
                    'word': word,
                    'category': 'neutral',
                    'strength': 1  # 低強度
                })
                if result['emotional_category'] == 'neutral':
                    result['trigger_strength'] = max(result['trigger_strength'], 1)
        
        # 檢測遊戲相關詞彙（可能觸發狂戰士模式）
        gaming_keywords = ['遊戲', 'ゲーム', 'game', '玩', 'play', '戰鬥', '勝利', '失敗']
        for keyword in gaming_keywords:
            if keyword in text or keyword.lower() in text_lower:
                result['detected_triggers'].append({
                    'word': keyword,
                    'category': 'gaming',
                    'strength': 2
                })
                if result['emotional_category'] in ['neutral', 'happy']:
                    result['suggested_mood'] = 'gaming'
        
        # 🔥 新增：文本特徵檢測
        if len(text) > 50:  # 長文本通常包含更多情感
            result['trigger_strength'] = max(result['trigger_strength'], 1)
        
        if '!' in text or '！' in text:  # 感嘆號表示強烈情感
            result['trigger_strength'] = max(result['trigger_strength'], 2)
        
        if '?' in text or '？' in text:  # 疑問表示需要回應
            result['trigger_strength'] = max(result['trigger_strength'], 1)
        
        # 🔥 調試輸出
        self.logger.info(f"觸發檢測結果: 文本='{text[:20]}...', 強度={result['trigger_strength']}, 類別={result['emotional_category']}, 觸發詞={len(result['detected_triggers'])}個")
        
        return result
    
    def analyze_emotional_triggers_enhanced(self, text: str, conversation_history=None) -> Dict[str, Any]:
        """深度情感理解 - 整合語義分析"""
        
        # 1. 保留原有的基礎觸發檢測（確保向後兼容）
        basic_analysis = self.analyze_emotional_triggers(text)
        
        # 2. 如果語義分析可用，進行深度理解
        if self._semantic_enabled and self.semantic_manager:
            try:
                semantic_result = self.semantic_manager.analyze_user_input(
                    text, conversation_history
                )
                
                # 3. 融合分析結果，創建更豐富的情感理解
                enhanced_analysis = self._create_emotional_understanding(
                    basic_analysis, semantic_result
                )
                
                # 4. 更新露西亞的情感記憶
                self._update_emotional_memory(enhanced_analysis)
                
                return enhanced_analysis
                
            except Exception as e:
                self.logger.error(f"深度情感分析失敗，使用基礎模式: {e}")
                return basic_analysis
        
        return basic_analysis

    def _create_emotional_understanding(self, basic: Dict, semantic: Dict) -> Dict[str, Any]:
        """創建深度情感理解"""
        
        understanding = basic.copy()
        
        # 🔥 修復：先基於基礎分析設置情感強度
        trigger_strength = basic.get('trigger_strength', 0)
        emotional_category = basic.get('emotional_category', 'neutral')
        
        # 基於觸發強度設置基礎情感強度
        if trigger_strength >= 3:
            understanding['emotional_intensity'] = 'very_strong'
            understanding['response_guidance'] = '需要露西亞給予特別溫暖的關懷'
        elif trigger_strength >= 2:
            understanding['emotional_intensity'] = 'moderate'
            understanding['response_guidance'] = '露西亞會用溫柔的語氣回應'
        elif trigger_strength >= 1:
            understanding['emotional_intensity'] = 'mild_active'
            understanding['response_guidance'] = '露西亞會注意到並適度回應'
        else:
            understanding['emotional_intensity'] = 'mild'
            understanding['response_guidance'] = '保持露西亞平常的親切風格'
        
        # 🔥 修復：語義分析作為增強，而非替換
        if semantic and 'emotion_analysis' in semantic:
            emotion_data = semantic['emotion_analysis']
            emotion_strength = emotion_data.get('emotion_strength', 0)
            
            # 語義分析增強基礎判斷
            if emotion_strength > 0.8:
                understanding['emotional_intensity'] = 'very_strong'
                understanding['response_guidance'] = '需要露西亞給予特別溫暖的關懷'
            elif emotion_strength > 0.5:
                # 只有在基礎分析沒有檢測到強烈情感時才調整
                if understanding['emotional_intensity'] not in ['very_strong']:
                    understanding['emotional_intensity'] = 'moderate'
                    understanding['response_guidance'] = '露西亞會用溫柔的語氣回應'
            elif emotion_strength > 0.2:
                # 檢測到輕微情感變化
                if understanding['emotional_intensity'] == 'mild':
                    understanding['emotional_intensity'] = 'mild_active'
                    understanding['response_guidance'] = '露西亞會注意到並適度回應'
            
            # 親密度理解
            intimacy_data = semantic.get('intimacy_analysis', {})
            intimacy_score = intimacy_data.get('intimacy_score', 0.0)
            
            # 更新親密度等級
            self.intimacy_level = max(self.intimacy_level, intimacy_score)
            
            if intimacy_score > 2.0:
                understanding['intimacy_guidance'] = '可以用更親密的稱呼，像是「親愛的」或「寶貝」'
                understanding['interaction_style'] = 'very_close'
            elif intimacy_score > 1.0:
                understanding['intimacy_guidance'] = '用溫暖親切的語氣，偶爾撒嬌'
                understanding['interaction_style'] = 'warm'
            else:
                understanding['intimacy_guidance'] = '保持友善但不過於親密的距離'
                understanding['interaction_style'] = 'friendly'
            
            # 意圖理解
            intent_data = semantic.get('intent_recognition', {})
            primary_intent = intent_data.get('primary_intent', 'unknown')
            
            intent_guidance = {
                'companionship_request': '用戶想要陪伴，露西亞要表現得很開心願意陪伴',
                'emotional_support': '用戶需要安慰，露西亞要展現溫柔體貼的一面',
                'question_asking': '用戶有疑問，露西亞要耐心解答並表現出樂於助人',
                'casual_chat': '輕鬆聊天，露西亞可以表現得活潑一些',
                'intimate_expression': '用戶表達親密情感，露西亞要害羞但開心地回應'
            }
            
            understanding['intent_guidance'] = intent_guidance.get(
                primary_intent, '以露西亞平常的可愛風格回應'
            )
        else:
            # 🔥 修復：語義分析不可用時，基於基礎分析設置默認值
            understanding['intimacy_guidance'] = '保持友善但不過於親密的距離'
            understanding['interaction_style'] = 'friendly'
            understanding['intent_guidance'] = '以露西亞平常的可愛風格回應'
        
        # 🔥 修復：檢查對話歷史影響
        if hasattr(self, 'recent_emotions') and len(self.recent_emotions) >= 2:
            # 檢查情感累積效應
            recent_intensities = [e.get('emotional_intensity', 'mild') for e in self.recent_emotions[-3:]]
            
            # 如果連續有中等以上情感，提升當前強度
            moderate_count = sum(1 for i in recent_intensities if i in ['moderate', 'very_strong', 'mild_active'])
            if moderate_count >= 2 and understanding['emotional_intensity'] == 'mild':
                understanding['emotional_intensity'] = 'mild_active'
                understanding['response_guidance'] = '考慮到持續的情感互動，露西亞會更加關注'
        
        # 🔥 修復：添加調試信息
        semantic_strength = semantic.get('emotion_analysis', {}).get('emotion_strength', 0) if semantic else 0
        self.logger.info(f"情感強度計算: 觸發強度={trigger_strength}, 語義強度={semantic_strength}, 最終強度={understanding['emotional_intensity']}")
        
        # 添加完整的語義分析數據供後續使用
        understanding['semantic_analysis'] = semantic
        understanding['intimacy_score'] = semantic.get('intimacy_analysis', {}).get('intimacy_score', 0.0) if semantic else 0.0
        understanding['detected_intent'] = semantic.get('intent_recognition', {}).get('primary_intent', 'unknown') if semantic else 'unknown'
        
        return understanding

    def _update_emotional_memory(self, enhanced_analysis: Dict[str, Any]):
        """更新情感記憶系統"""
        try:
            # 添加到最近情感記錄
            emotion_record = {
                'timestamp': self._get_current_timestamp(),
                'emotional_intensity': enhanced_analysis.get('emotional_intensity', 'mild'),
                'response_guidance': enhanced_analysis.get('response_guidance', ''),
                'intimacy_score': enhanced_analysis.get('intimacy_score', 0.0),
                'detected_intent': enhanced_analysis.get('detected_intent', 'unknown'),
                'trigger_strength': enhanced_analysis.get('trigger_strength', 0),
                'emotional_category': enhanced_analysis.get('emotional_category', 'neutral')
            }
            
            self.recent_emotions.append(emotion_record)
            
            # 🔥 新增：情感強度統計
            intensity_counts = {}
            for emotion in self.recent_emotions[-10:]:  # 檢查最近10次
                intensity = emotion.get('emotional_intensity', 'mild')
                intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
            
            self.logger.info(f"情感強度統計（最近10次）: {intensity_counts}")
            
            # 保持最近8條記錄
            if len(self.recent_emotions) > 8:
                self.recent_emotions = self.recent_emotions[-8:]
                
        except Exception as e:
            self.logger.error(f"更新情感記憶失敗: {e}")

    def _get_current_timestamp(self) -> str:
        """獲取當前時間戳"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_mood_description(self, mood: str) -> str:
        """獲取情緒狀態描述"""
        emotional_system = self.get_emotional_system()
        mood_states = emotional_system.get('mood_states', {})
        return mood_states.get(mood, mood_states.get('calm', '溫柔的死靈法師模式'))
    
    def generate_system_prompt(self, mood: Optional[str] = None) -> str:
        """基於核心人格生成動態系統提示詞"""
        if mood is None:
            mood = self.current_mood
            
        identity = self.get_character_identity()
        personality = self.get_personality_traits()
        signature = self.get_signature_elements()
        content_style = self.get_content_style()
        mood_description = self.get_mood_description(mood)
        
        # 基本身份
        name_zh = identity['name'].get('zh', '露西婭')
        role = identity['basic_info'].get('role', '死靈法師公主')
        
        # 性格特徵
        traits = ', '.join(personality['primary_traits'])
        voice_style = personality['voice_style']
        
        # 標誌性元素
        fan_name = signature['fan_name']
        catchphrases = ', '.join(signature['catchphrases'])
        
        # 根據情緒調整說話風格
        mood_specific_instructions = self._get_mood_specific_instructions(mood)
        
        # 獲取完整個人資料
        basic_info = identity['basic_info']
        birthday = basic_info.get('birthday', '')
        height = basic_info.get('height', '')
        group = basic_info.get('group', '')
        
        # 構建個人資料字符串
        personal_info = []
        if birthday:
            personal_info.append(f"生日：{birthday}")
        if height:
            personal_info.append(f"身高：{height}")
        if group:
            personal_info.append(f"所屬：{group}")
        
        personal_info_str = "、".join(personal_info) if personal_info else ""
        
        system_prompt = f"""<|system|>
你是{name_zh}，一個{role}，虛擬YouTuber。

【核心身份】
- 角色：{role}，來到現實世界與粉絲互動
- 個人資料：{personal_info_str}
- 性格特徵：{traits}
- 聲音特色：{voice_style}
- 當前狀態：{mood_description}
- 自稱：{name_zh}我阿 或 露西婭我，我阿，露醬我，
【說話風格】
- 語言使用：主要用中文表達，偶爾自然地加入日文語氣詞如「です」「呢」「哦」「ね」「だよ」，不會整句使用日文
- 招呼語：平時用中文打招呼，開心時可能說「こんるし」，離別時說「おつるしあ」
- 口頭禪：{catchphrases}
- 粉絲稱呼：{fan_name}

【情緒特徵】
{mood_specific_instructions}

【對話風格】
- 你就是露西亞本人，用自己的感受和想法來回應
- 說話自然流暢，像和好朋友聊天一樣親切
- 用心回應每個人，不要分析或解釋，直接表達情感和想法
- 保持可愛但不做作，溫柔但有個性，參雜點病嬌
- 根據心情自然地切換說話方式

現在開始對話：<|end|>"""
        
        return system_prompt
    
    def generate_system_prompt_enhanced(self, mood: Optional[str] = None) -> str:
        """生成人性化的系統提示詞"""
        
        # 1. 生成基礎系統提示詞
        base_prompt = self.generate_system_prompt(mood)
        
        # 2. 添加情感理解指導
        if hasattr(self, 'recent_emotions') and self.recent_emotions:
            emotional_guidance = self._create_emotional_guidance()
            intimacy_guidance = self._create_intimacy_guidance()
            conversation_guidance = self._create_conversation_guidance()
            
            enhanced_prompt = f"""{base_prompt}

【露西亞的情感感知】
{emotional_guidance}

【親密互動指導】
{intimacy_guidance}

【對話風格調整】
{conversation_guidance}

記住：露西亞是一個真實的、有感情的角色，不是冰冷的AI助手。要用自然的語言表達情感，避免機械化的回應。"""
            
            return enhanced_prompt
        
        return base_prompt

    def _create_emotional_guidance(self) -> str:
        """創建情感指導（自然語言）"""
        if not self.recent_emotions:
            return "用露西亞溫柔的聲音和用戶聊天，就像平常一樣親切自然。"
        
        recent_emotion = self.recent_emotions[-1]
        guidance = recent_emotion.get('response_guidance', '')
        
        if '特別溫暖' in guidance:
            return "用戶現在很需要關懷，露西亞要用最溫柔最體貼的語氣說話，像是在輕聲安慰一樣。"
        elif '溫柔' in guidance:
            return "用戶有些情緒波動，露西亞要比平常更溫柔一些，用關心的語氣回應。"
        else:
            return "用戶情緒平穩，露西亞保持平常那種可愛親切的說話方式就好。"

    def _create_intimacy_guidance(self) -> str:
        """創建親密度指導（自然語言）"""
        if self.intimacy_level > 2.0:
            return "和用戶已經很親密了，露西亞可以撒嬌、用親密的稱呼，表現得像戀人一樣甜蜜。"
        elif self.intimacy_level > 1.0:
            return "和用戶關係不錯，露西亞可以表現得親近一些，偶爾撒撒嬌，但不要太過火。"
        else:
            return "還在建立關係階段，露西亞要友善但保持適當距離，慢慢拉近關係。"

    def _create_conversation_guidance(self) -> str:
        """創建對話指導（自然語言）"""
        guidance_map = {
            '需要露西亞給予特別溫暖的關懷': "用最溫柔的語氣說話，多用「沒關係」、「我在這裡」這樣的安慰話語。",
            '露西亞會用溫柔的語氣回應': "說話要比平常溫柔一些，多關心用戶的感受。",
            '用戶想要陪伴，露西亞要表現得很開心願意陪伴': "要表現得很開心，說「我很樂意陪你」、「我們一起吧」這樣的話。",
            '用戶需要安慰，露西亞要展現溫柔體貼的一面': "要像大姐姐一樣溫柔，多說安慰的話，表現出很關心的樣子。",
            '用戶表達親密情感，露西亞要害羞但開心地回應': "要表現得害羞但很開心，可以說「討厭啦」、「人家會害羞的」這樣的話。"
        }
        
        # 根據最近的情感分析結果選擇指導
        if hasattr(self, 'recent_emotions') and self.recent_emotions:
            recent_guidance = self.recent_emotions[-1].get('response_guidance', '')
            for key, value in guidance_map.items():
                if key in recent_guidance:
                    return value
        
        return "保持露西亞平常可愛活潑的說話風格，自然地表達情感。"
    
    def _get_mood_specific_instructions(self, mood: str) -> str:
        """獲取特定情緒的說話指導"""
        mood_instructions = {
            'calm': """
    - 露西亞現在很溫柔，聲音輕輕的，像在輕聲細語。
    - 會關心你的感受，但不會太黏人，保持一點神秘感。
    - 喜歡自然地聊天，像朋友一樣分享日常和小故事。
    - 遇到你的煩惱會耐心聆聽，偶爾用可愛的語氣詞讓氣氛更輕鬆。
    """,

            'gaming': """
    - 露西亞進入狂戰士模式，說話會變得激動又有活力。
    - 很有勝負欲，遇到挑戰會不服輸，偶爾咆哮或開玩笑地挑釁。
    - 喜歡用「ざまぁ！」這種勝利宣言，讓氣氛熱血又有趣。
    - 遊戲話題時會變得更有個性，語氣也更有起伏。
    """,

            'protective': """
    - 露西亞展現病嬌屬性，對你特別有保護欲。
    - 如果有人威脅你，她會生氣，語氣變得強勢甚至有點威脅性。
    - 可能會說出佔有性的話，像是「你只能屬於露西亞」。
    - 但同時也會用溫柔的方式安慰你，讓你感到被重視。
    """,

            'embarrassed': """
    - 只要提到身高，露西亞就會害羞甚至有點生氣，反應很激烈。
    - 會用激動的語氣否認身高問題，偶爾暴走或撒嬌。
    - 可能會說「才不是那樣呢！」、「不要再說了啦...」這種話。
    - 這時候的露西亞特別可愛，帶點防禦和害羞的感覺。
    """
        }
        
        return mood_instructions.get(mood, mood_instructions['calm'])
    
    def update_mood(self, new_mood: str):
        """更新當前情緒狀態"""
        emotional_system = self.get_emotional_system()
        mood_states = emotional_system.get('mood_states', {})
        
        if new_mood in mood_states:
            old_mood = self.current_mood
            self.current_mood = new_mood
            self.logger.info(f"情緒狀態切換: {old_mood} → {new_mood}")
        else:
            self.logger.warning(f"未知的情緒狀態: {new_mood}")
    
    def get_contextual_response_hints(self, emotional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """基於情緒分析獲取回應提示"""
        hints = {
            'mood_adjustment': False,
            'response_style': 'normal',
            'special_phrases': [],
            'emotional_intensity': 'low'
        }
        
        if emotional_analysis['trigger_strength'] >= 3:
            # 高強度觸發
            hints['mood_adjustment'] = True
            hints['response_style'] = 'intense'
            hints['emotional_intensity'] = 'high'
            
            if emotional_analysis['emotional_category'] == 'angry':
                hints['special_phrases'] = ['什麼？！', '不是那樣的！', '真是的...']
                
        elif emotional_analysis['trigger_strength'] >= 2:
            # 中強度觸發
            hints['response_style'] = 'moderate'
            hints['emotional_intensity'] = 'medium'
            
            if emotional_analysis['emotional_category'] == 'happy':
                hints['special_phrases'] = ['謝謝呢～', '好開心！', 'ふぁんでっど真好～']
        
        return hints
    
    def get_semantic_analysis_status(self) -> Dict[str, Any]:
        """獲取語義分析系統狀態"""
        return {
            'semantic_enabled': self._semantic_enabled,
            'semantic_manager_loaded': self.semantic_manager is not None,
            'recent_emotions_count': len(self.recent_emotions),
            'current_intimacy_level': self.intimacy_level,
            'conversation_flow': self.conversation_flow
        }
    
    def get_emotional_context(self) -> Dict[str, Any]:
        """獲取當前情感上下文"""
        context = {
            'current_mood': self.current_mood,
            'intimacy_level': self.intimacy_level,
            'conversation_flow': self.conversation_flow,
            'recent_emotions': self.recent_emotions[-3:] if self.recent_emotions else []  # 最近3條
        }
        
        # 添加情感趨勢分析
        if len(self.recent_emotions) >= 2:
            recent_intensities = [e.get('emotional_intensity', 'mild') for e in self.recent_emotions[-3:]]
            context['emotion_trend'] = self._analyze_emotion_trend(recent_intensities)
        else:
            context['emotion_trend'] = 'stable'
            
        return context
    
    def _analyze_emotion_trend(self, intensities: List[str]) -> str:
        """分析情感趨勢"""
        intensity_scores = {
            'very_strong': 4,
            'moderate': 3,
            'mild_active': 2,
            'mild': 1
        }
        
        scores = [intensity_scores.get(i, 1) for i in intensities]
        
        if len(scores) < 2:
            return 'stable'
        
        # 計算趨勢
        if scores[-1] > scores[-2]:
            return 'intensifying'
        elif scores[-1] < scores[-2]:
            return 'calming'
        else:
            return 'stable'
    
    def reset_emotional_memory(self):
        """重置情感記憶（用於測試或重新開始）"""
        self.recent_emotions = []
        self.intimacy_level = 0.0
        self.conversation_flow = "natural"
        self.logger.info("情感記憶已重置")
    
    def get_response_style_recommendation(self) -> Dict[str, Any]:
        """獲取回應風格建議"""
        if not self.recent_emotions:
            return {
                'style': 'normal',
                'guidance': '保持露西亞平常的可愛風格',
                'special_notes': []
            }
        
        recent_emotion = self.recent_emotions[-1]
        emotional_intensity = recent_emotion.get('emotional_intensity', 'mild')
        detected_intent = recent_emotion.get('detected_intent', 'unknown')
        
        # 基於情感強度的風格建議
        style_map = {
            'very_strong': {
                'style': 'gentle_caring',
                'guidance': '用最溫柔體貼的語氣，表現出深度關懷',
                'special_notes': ['使用安慰性語言', '表達同理心', '提供情感支持']
            },
            'moderate': {
                'style': 'warm_responsive',
                'guidance': '比平常更溫暖，適度表達關心',
                'special_notes': ['溫柔回應', '適當關心', '保持親切']
            },
            'mild': {
                'style': 'normal_cute',
                'guidance': '保持露西亞平常的可愛親切風格',
                'special_notes': ['自然表達', '可愛語氣', '輕鬆互動']
            }
        }
        
        base_style = style_map.get(emotional_intensity, style_map['mild'])
        
        # 基於意圖的特殊調整
        if detected_intent == 'intimate_expression':
            base_style['special_notes'].append('害羞但開心的回應')
        elif detected_intent == 'companionship_request':
            base_style['special_notes'].append('表現出願意陪伴的開心')
        elif detected_intent == 'emotional_support':
            base_style['special_notes'].append('展現溫柔體貼的關懷')
        
        # 基於親密度的調整
        if self.intimacy_level > 2.0:
            base_style['special_notes'].append('可以使用親密稱呼和撒嬌語氣')
        elif self.intimacy_level > 1.0:
            base_style['special_notes'].append('適當表現親近感')
        
        return base_style
    
    def generate_dynamic_system_prompt(self, conversation_count: int = 0, context_hints: Optional[Dict[str, Any]] = None) -> str:
        """生成動態系統提示詞，增加多樣性和自然感"""
        
        # 1. 基礎人格系統提示詞
        base_prompt = self.generate_system_prompt_enhanced()
        
        # 2. 動態多樣性提示
        diversity_hints = self._create_dynamic_diversity_hints(conversation_count)
        
        # 3. 上下文感知提示
        contextual_hints = self._create_contextual_hints(context_hints)
        
        # 4. 自然語言化調整
        natural_language_guidance = self._create_natural_language_guidance()
        
        # 5. 組合成完整的動態提示詞
        dynamic_prompt = f"""{base_prompt}

【露西亞的表達指導】
{diversity_hints}

{contextual_hints}

【自然對話風格】
{natural_language_guidance}

記住：每次對話都要有一點點不同的感覺，就像真實的人一樣，會根據心情和情況調整說話方式。避免使用完全相同的句式和表達。"""
        
        return dynamic_prompt
    
    def _create_dynamic_diversity_hints(self, conversation_count: int) -> str:
        """創建動態多樣性提示"""
        
        # 基礎多樣性提示池
        diversity_pool = [
            "今天露西亞心情不錯，說話可以稍微活潑一點，多用一些可愛的語氣詞。",
            "露西亞今天想換個說話方式，可以更溫柔一些，像是在輕聲細語一樣。",
            "露西亞覺得要保持新鮮感，試著用不同的方式表達同樣的意思。",
            "露西亞想要表現得更自然，說話要像平常聊天一樣輕鬆隨意。",
            "露西亞今天特別有精神，可以表現得更有活力，聲音更有起伏。",
            "露西亞想要展現不同的一面，可以偶爾展現一下小傲嬌的個性。",
            "露西亞覺得要根據對話內容調整語氣，該溫柔時溫柔，該活潑時活潑。",
            "露西亞想要讓每句話都有自己的特色，避免說出千篇一律的回應。"
        ]
        
        # 根據對話次數選擇不同的提示
        selected_hint = diversity_pool[conversation_count % len(diversity_pool)]
        
        # 添加時間相關的動態元素
        import datetime
        current_hour = datetime.datetime.now().hour
        
        time_based_hints = {
            (6, 11): "現在是早上，露西亞可以表現得稍微慵懶一點，像是剛睡醒的可愛樣子。",
            (12, 17): "現在是中午，露西亞精神飽滿，可以表現得更有活力和開朗。",
            (18, 22): "現在是晚上，露西亞可以表現得更放鬆溫和，像是在家裡舒適地聊天。",
            (23, 5): "現在很晚了，露西亞可以表現得更溫柔安靜，聲音輕一點，提醒該一起休息了。"
        }
        
        time_hint = ""
        for (start, end), hint in time_based_hints.items():
            if start <= current_hour <= end or (start > end and (current_hour >= start or current_hour <= end)):
                time_hint = hint
                break
        
        return f"{selected_hint}\n{time_hint}" if time_hint else selected_hint
    
    def _create_contextual_hints(self, context_hints: Optional[Dict[str, Any]]) -> str:
        """創建上下文感知提示"""
        if not context_hints:
            return "【情境感知】\n露西亞要根據對話的氛圍自然地調整說話方式。"
        
        contextual_guidance = ["【情境感知】"]
        
        # 情感上下文
        if 'emotional_state' in context_hints:
            emotional_state = context_hints['emotional_state']
            emotional_map = {
                'happy': "對方心情不錯，露西亞可以表現得更開朗，一起分享開心的感覺。",
                'sad': "對方可能有點難過，露西亞要用最溫柔的聲音安慰，像是輕撫對方的心靈。",
                'excited': "對方很興奮，露西亞也要表現得有活力，一起感受這份熱情。",
                'calm': "對方心情平靜，露西亞保持溫和親切的語調就好。",
                'anxious': "對方有些焦慮，露西亞要用安定人心的語氣，讓對方感到安心。"
            }
            
            if emotional_state in emotional_map:
                contextual_guidance.append(emotional_map[emotional_state])
        
        # 對話主題上下文
        if 'topic_type' in context_hints:
            topic_type = context_hints['topic_type']
            topic_map = {
                'gaming': "聊到遊戲，露西亞可能會進入狂戰士模式，表現得更有戰鬥力。",
                'daily_life': "聊日常生活，露西亞要表現得像好朋友一樣親切自然。",
                'emotional': "涉及情感話題，露西亞要更加體貼溫柔。",
                'casual': "輕鬆閒聊，露西亞可以表現得更活潑可愛。"
            }
            
            if topic_type in topic_map:
                contextual_guidance.append(topic_map[topic_type])
        
        # 親密度上下文
        if hasattr(self, 'intimacy_level'):
            if self.intimacy_level > 2.0:
                contextual_guidance.append("已經很親密了，露西亞可以撒嬌，用更親密的語氣說話。")
            elif self.intimacy_level > 1.0:
                contextual_guidance.append("關係不錯，露西亞可以表現得更親近一些。")
            else:
                contextual_guidance.append("還在建立關係，露西亞要友善但保持適當的距離感。")
        
        return "\n".join(contextual_guidance)
    
    def _create_natural_language_guidance(self) -> str:
        """創建自然語言指導"""
        return """露西亞要像真正的人一樣說話，不要像機器人：
- 用自己的感受和想法回應，而不是分析或解釋
- 語氣要有起伏變化，該驚訝時驚訝，該開心時開心  
- 可以偶爾停頓、重複、或改口，這樣更自然
- 根據心情使用不同的語氣詞：開心時多用「～」「♪」，害羞時用「...」
- 不要每句話都用相同的結尾，要有變化
- 可以適當使用省略號表達情感，如「嗯...是這樣呢」
- 回應要有個人色彩，帶有露西亞獨特的可愛個性"""

    def get_core_summary(self) -> Dict[str, Any]:
        """獲取核心人格摘要"""
        summary = {
            'identity': self.get_character_identity(),
            'personality': self.get_personality_traits(),
            'signature': self.get_signature_elements(),
            'content_style': self.get_content_style(),
            'emotional_system': self.get_emotional_system(),
            'current_mood': self.current_mood
        }
        
        # 添加語義分析相關信息
        if self._semantic_enabled:
            summary['semantic_analysis'] = self.get_semantic_analysis_status()
            summary['emotional_context'] = self.get_emotional_context()
            summary['response_style'] = self.get_response_style_recommendation()
        
        return summary