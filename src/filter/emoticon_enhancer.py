#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顏文字表情增強過濾器
專門添加可愛的顏文字表情，提升回應的表現力和親近感
"""

import re
import logging
import random
from typing import Dict, Any, Optional, Tuple, List
from .base_filter import BaseResponseFilter

logger = logging.getLogger(__name__)

class EmoticonEnhancerFilter(BaseResponseFilter):
    """顏文字表情增強過濾器"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 初始化顏文字庫
        self._initialize_emoticons()
        
        logger.info("顏文字表情增強過濾器初始化完成")
    
    def _initialize_emoticons(self):
        """初始化顏文字表情庫"""
        
        # 問候/打招呼顏文字
        self.greeting_emoticons = [
            "(•̀ω•́)✧",
            "（＾ω＾）",
            "(´∀｀)",
            "ヽ(✿ﾟ▽ﾟ)ノ",
            "(｡◕‿◕｡)",
            "（＾▽＾）",
            "(⌒‿⌒)",
            "ヾ(＾-＾)ノ",
            "( ´ ▽ ` )",
            "(◕‿◕)♡"
        ]
        
        # 開心/興奮顏文字
        self.happy_emoticons = [
            "＼(^o^)／",
            "ヽ(°〇°)ﾉ",
            "(ﾉ◕ヮ◕)ﾉ",
            "＼(^▽^)／",
            "ヾ(≧▽≦*)o",
            "(ﾉ*°▽°*)ﾉ",
            "(*^▽^*)",
            "o(≧▽≦)o",
            "ヽ(´▽`)ﾉ",
            "(◕‿◕)♪"
        ]
        
        # 溫柔/愛意顏文字
        self.gentle_emoticons = [
            "(´∀｀)",
            "(*´∀｀*)",
            "(｡♥‿♥｡)",
            "(◕‿◕)♡",
            "(*°∀°)=3",
            "(⁎⁍̴̛ᴗ⁍̴̛⁎)",
            "( ˘▾˘)~♪",
            "(◡ ω ◡)",
            "( ´ ∀ ` )",
            "(*´ω｀*)"
        ]
        
        # 害羞/可愛顏文字
        self.shy_emoticons = [
            "(//▽//)",
            "(⁄ ⁄•⁄ω⁄•⁄ ⁄)",
            "(*ﾟ∀ﾟ*)",
            "(⌒_⌒;)",
            "( ˶ᵔ ᵕ ᵔ˶ )",
            "(˶ᵔ ᵕ ᵔ˶)",
            "(*ﾟ▽ﾟ*)",
            "( ¨̮ )",
            "(˘▾˘)~",
            "( ◡‿◡ )"
        ]
        
        # 關心/安慰顏文字
        self.caring_emoticons = [
            "(´･ω･`)",
            "(｡◕‿◕｡)",
            "( ˘ ³˘)♥",
            "(っ˘̩╭╮˘̩)っ",
            "(´∀｀)♡",
            "( ˘▾˘)~",
            "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧",
            "ヾ(•ω•`)o",
            "( ˶ᵔ ᵕ ᵔ˶ )",
            "(*´∀｀)ノ"
        ]
        
        # 疑問/好奇顏文字
        self.curious_emoticons = [
            "(・∀・)?",
            "(｡◕‿‿◕｡)?",
            "(´･ω･`)?",
            "( ゜o゜)",
            "Σ(゜o゜)",
            "(◯ᵔ＿ᵔ◯)",
            "(｡•́‿•̀｡)?",
            "( ´∀｀)?",
            "(◉‿◉)?",
            "(._.?)？"
        ]
        
        # 撒嬌/親密顏文字
        self.intimate_emoticons = [
            "(｡♥‿♥｡)",
            "( ˘ ³˘)♥",
            "(っ˘̩╭╮˘̩)っ",
            "(*´∀｀)ﾉ",
            "(´∀｀)♡",
            "( ◡‿◡ )♡",
            "(*°▽°*)",
            "(⁎⁍̴̛ᴗ⁍̴̛⁎)",
            "( ˶ᵔ ᵕ ᵔ˶ )♡",
            "(˶ˆ꒳ˆ˵)"
        ]
        
        # 特殊場景顏文字 - 根據關鍵詞觸發
        self.scenario_emoticons = {
            # 食物相關
            'food': ["(๑´ڡ`๑)", "(*´﹃｀*)", "(っ˘ڡ˘ς)", "( ˘▾˘)~♪", "ლ(◕ω◕ლ)", "(｡•̀ᴗ-)✧"],
            # 睡眠相關
            'sleep': ["(-ω-)", "(｡-ω-)zzz", "( ˘ω˘ )ｽﾔｧ", "(｡-_-｡)", "( ˘▾˘)~"],
            # 音樂相關
            'music': ["♪( ´▽｀)", "♪(´ε｀ )", "ヽ(´▽｀)ノ♪", "♪ヽ(´▽`)ﾉ", "( ˘▾˘)~♪"],
            # 遊戲相關
            'game': ["ヽ(°〇°)ﾉ", "＼(^o^)／", "(ﾉ◕ヮ◕)ﾉ", "ヾ(≧▽≦*)o", "o(≧▽≦)o"],
            # 學習相關
            'study': ["(｡◕‿◕｡)", "ヾ(•ω•`)o", "( ´∀｀)", "(*´∀｀)", "(◕‿◕)✧"],
            # 早安/問候特殊
            'morning': ["(•̀ω•́)✧", "（^▽^）", "ヽ(✿ﾟ▽ﾟ)ノ", "(⌒‿⌒)", "( ´ ▽ ` )"],
            # 親密抱抱
            'hug': ["(´∀｀)", "(っ˘̩╭╮˘̩)っ", "( ˘ ³˘)♥", "(⁎⁍̴̛ᴗ⁍̴̛⁎)", "( ◡‿◡ )♡"]
        }
        
        # 編譯關鍵詞模式
        self._compile_keyword_patterns()
    
    def _compile_keyword_patterns(self):
        """編譯關鍵詞匹配模式"""
        
        # 問候語相關關鍵詞
        self.greeting_keywords = [
            '早安', '午安', '晚安', '你好', '嗨', 'hello', 'hi', '哈囉', 
            '回來', '歡迎', '初次見面', '好久不見'
        ]
        
        # 開心情緒關鍵詞
        self.happy_keywords = [
            '開心', '高興', '快樂', '興奮', '棒', '太好了', '厲害', '讚', 
            '超棒', '很棒', 'happy', 'great', 'awesome', '嬉しい'
        ]
        
        # 害羞情境關鍵詞
        self.shy_keywords = [
            '害羞', '不好意思', '臉紅', '可愛', '美麗', '漂亮', 
            '稱讚', '誇獎', '謝謝', 'cute', 'beautiful', 'pretty'
        ]
        
        # 溫柔情境關鍵詞
        self.gentle_keywords = [
            '溫柔', '溫暖', '舒服', '放鬆', '安心', '平靜', '柔軟', 
            '輕柔', '暖暖', '溫馨', '甜蜜', '感謝'
        ]
        
        # 關心情境關鍵詞
        self.caring_keywords = [
            '擔心', '關心', '照顧', '安慰', '沒關係', '辛苦', '累', 
            '休息', '保重', '加油', '支持'
        ]
        
        # 疑問情境關鍵詞
        self.curious_keywords = [
            '什麼', '為什麼', '怎麼', '哪裡', '誰', '嗎', '呢', 
            'what', 'why', 'how', 'where', 'who'
        ]
        
        # 親密情境關鍵詞
        self.intimate_keywords = [
            '愛', '喜歡', '想你', '思念', '陪伴', '一起', '抱抱', 
            '親親', '撒嬌', '寵愛', 'love', '好き', '大好き'
        ]
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """添加顏文字表情"""
        if not response:
            return response
        
        # 檢查是否需要添加顏文字
        if not self._should_add_emoticon(response, user_input):
            return response
        
        original_response = response
        
        try:
            # 根據情境選擇合適的顏文字
            emoticon = self._select_emoticon(response, user_input)
            
            if emoticon:
                # 智慧插入顏文字
                response = self._insert_emoticon(response, emoticon, user_input)
            
            logger.debug(f"顏文字增強: {original_response[:30]}... -> {response[:30]}...")
            
        except Exception as e:
            logger.error(f"顏文字增強失敗: {e}")
            return original_response
        
        return response
    
    def _should_add_emoticon(self, response: str, user_input: str) -> bool:
        """判斷是否需要添加顏文字"""
        
        # 如果回應已經有很多表情符號，就不添加顏文字
        existing_symbols = response.count('♪') + response.count('♡') + response.count('～')
        existing_emoticons = len(re.findall(r'\([^)]{2,8}\)', response))
        
        if existing_symbols >= 3 or existing_emoticons >= 1:
            return False
        
        # 如果回應太短，添加顏文字會顯得突兀
        if len(response.strip()) < 8:
            return False
        
        # 如果回應太長，選擇性添加
        if len(response) > 50 and random.random() < 0.3:
            return False
        
        # 檢查是否有適合的情境
        has_suitable_context = (
            (user_input and any(keyword in user_input for keyword in 
                self.greeting_keywords + self.happy_keywords + 
                self.gentle_keywords + self.intimate_keywords)) or
            any(keyword in response for keyword in 
                ['開心', '高興', '溫柔', '可愛', '謝謝', '你好'])
        )
        
        return has_suitable_context or random.random() < 0.4  # 40% 基礎機率
    
    def _select_emoticon(self, response: str, user_input: str) -> Optional[str]:
        """根據情境選擇合適的顏文字"""
        
        # 分析用戶輸入的情感
        user_emotion = self._analyze_emotion(user_input)
        response_emotion = self._analyze_emotion(response)
        
        # 優先根據用戶情感選擇
        primary_emotion = user_emotion if user_emotion else response_emotion
        
        # 根據主要情感選擇顏文字庫
        if primary_emotion == 'greeting':
            emoticons = self.greeting_emoticons
        elif primary_emotion == 'happy':
            emoticons = self.happy_emoticons
        elif primary_emotion == 'gentle':
            emoticons = self.gentle_emoticons
        elif primary_emotion == 'shy':
            emoticons = self.shy_emoticons
        elif primary_emotion == 'caring':
            emoticons = self.caring_emoticons
        elif primary_emotion == 'curious':
            emoticons = self.curious_emoticons
        elif primary_emotion == 'intimate':
            emoticons = self.intimate_emoticons
        else:
            # 如果沒有明確情感，根據特殊場景選擇
            scene_emoticon = self._select_scene_emoticon(user_input + " " + response)
            if scene_emoticon:
                return scene_emoticon
            # 否則使用溫柔表情作為默認
            emoticons = self.gentle_emoticons
        
        return random.choice(emoticons)
    
    def _analyze_emotion(self, text: str) -> Optional[str]:
        """分析文本中的主要情感"""
        text_lower = text.lower()
        
        # 按優先級檢查各種情感
        if any(keyword in text_lower for keyword in self.greeting_keywords):
            return 'greeting'
        elif any(keyword in text_lower for keyword in self.happy_keywords):
            return 'happy'
        elif any(keyword in text_lower for keyword in self.intimate_keywords):
            return 'intimate'
        elif any(keyword in text_lower for keyword in self.shy_keywords):
            return 'shy'
        elif any(keyword in text_lower for keyword in self.caring_keywords):
            return 'caring'
        elif any(keyword in text_lower for keyword in self.curious_keywords):
            return 'curious'
        elif any(keyword in text_lower for keyword in self.gentle_keywords):
            return 'gentle'
        
        return None
    
    def _select_scene_emoticon(self, text: str) -> Optional[str]:
        """根據特殊場景選擇顏文字"""
        text_lower = text.lower()
        
        # 早安特殊處理
        morning_words = ['早安', '早上好', 'good morning', 'おはよう']
        if any(word in text_lower for word in morning_words):
            return random.choice(self.scenario_emoticons['morning'])
        
        # 親密抱抱場景
        hug_words = ['抱', '擁抱', '依偎', '懷裡', '躺下', 'hug', '緊緊', '靠近']
        if any(word in text_lower for word in hug_words):
            return random.choice(self.scenario_emoticons['hug'])
        
        # 食物場景
        food_words = ['吃', '食物', '美食', '料理', '甜品', '蛋糕', '餅乾', '美味', '好吃']
        if any(word in text_lower for word in food_words):
            return random.choice(self.scenario_emoticons['food'])
        
        # 睡眠場景
        sleep_words = ['睡', '休息', '累', '疲倦', '困', '晚安', '好夢']
        if any(word in text_lower for word in sleep_words):
            return random.choice(self.scenario_emoticons['sleep'])
        
        # 音樂場景
        music_words = ['音樂', '歌', '唱', '聽', 'music', 'song', '旋律', '節奏']
        if any(word in text_lower for word in music_words):
            return random.choice(self.scenario_emoticons['music'])
        
        # 遊戲場景
        game_words = ['遊戲', '玩', '電玩', 'game', '競賽', '比賽', '挑戰']
        if any(word in text_lower for word in game_words):
            return random.choice(self.scenario_emoticons['game'])
        
        # 學習場景
        study_words = ['學習', '讀書', '研究', '知識', '學', '教', '了解']
        if any(word in text_lower for word in study_words):
            return random.choice(self.scenario_emoticons['study'])
        
        return None
    
    def _insert_emoticon(self, response: str, emoticon: str, user_input: str) -> str:
        """智慧插入顏文字到合適的位置"""
        
        # 如果是問候語，放在開頭
        if user_input and any(keyword in user_input.lower() for keyword in self.greeting_keywords):
            return f"{emoticon}{response}"
        
        # 如果回應較短，放在開頭
        if len(response) < 20:
            return f"{emoticon}{response}"
        
        # 如果回應較長，智慧選擇位置
        # 優先放在句子的自然分割點
        sentences = re.split(r'[。！？♪♡～]', response)
        if len(sentences) > 1 and len(sentences[0]) > 5:
            # 在第一句話後插入
            first_sentence_end = len(sentences[0])
            for i, char in enumerate(response):
                if i >= first_sentence_end and char in '。！？♪♡～':
                    return response[:i+1] + emoticon + response[i+1:]
        
        # 如果找不到合適位置，放在開頭
        return f"{emoticon}{response}"
    
    def get_filter_name(self) -> str:
        """回傳過濾器名稱"""
        return "EmoticonEnhancer"
    
    def get_filter_description(self) -> str:
        """回傳過濾器描述"""
        return "顏文字表情增強：根據情境智慧添加可愛的顏文字表情"
