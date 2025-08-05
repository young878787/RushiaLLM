#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
語義關鍵詞配置
集中管理所有語義分析相關的關鍵詞和配置
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class SemanticKeywordConfig:
    """語義關鍵詞配置類"""
    
    def __init__(self):
        # 語義關鍵詞庫 - 從主程式遷移
        self.semantic_keywords = {
            'emotion_positive': ['開心', '高興', '快樂', '幸福', '喜歡', '愛', '喜悅', '滿足', '溫暖', '甜蜜', '好'],
            'emotion_negative': ['難過', '傷心', '生氣', '煩躁', '沮喪', '失望', '痛苦', '焦慮', '害怕', '孤單', '寂寞', '累'],
            'intimacy_high': ['愛你', '親親', '抱抱', '一起', '陪伴', '想你', '思念', '溫柔', '甜蜜', '心跳', '牽手', '帶你', '陪你'],
            'intimacy_medium': ['喜歡', '關心', '溫暖', '舒服', '安心', '依賴', '信任', '珍惜', '聊天'],
            'question_words': ['什麼', '哪裡', '怎麼', '為什麼', '誰', '何時', '如何', '可以嗎', '好嗎', '想吃什麼'],
            'action_words': ['做', '在做', '準備', '想要', '希望', '打算', '計劃', '開始', '結束', '帶', '去', '走', '來'],
            'time_words': ['現在', '今天', '明天', '昨天', '早上', '中午', '晚上', '最近', '之前', '以後'],
            'food_words': ['吃', '餓', '食物', '料理', '美食', '飯', '菜', '漢堡', '早餐', '午餐', '晚餐'],
            'greeting_words': ['你好', '中午好', '早安', '晚安', 'hello', 'hi', '午安', '早上好', '晚上好'],
            'companionship_words': ['陪', '一起', '帶', '跟', '和', '同', '共同', '陪伴', '牽手', '牽著']
        }
        
        # 情感權重計算
        self.emotion_weights = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
        
        # 親密度計算因子
        self.intimacy_factors = {
            'physical_contact': 2.0,  # 身體接觸類
            'emotional_expression': 1.5,  # 情感表達類
            'companionship': 1.0,  # 陪伴類
            'general_affection': 0.5  # 一般喜愛類
        }
        
        # 話題分類映射
        self.topic_mapping = {
            'food': self.semantic_keywords['food_words'],
            'greeting': self.semantic_keywords['greeting_words'],
            'time_aware': self.semantic_keywords['time_words'],
            'intimate': self.semantic_keywords['intimacy_high'] + self.semantic_keywords['intimacy_medium'],
            'emotional_support': ['累', '疲勞', '難過', '傷心', '沮喪', '壓力', '煩惱', '寂寞', '孤單'],
            'companionship': self.semantic_keywords['companionship_words']
        }
        
        # 自定義詞庫（用於jieba）
        self.custom_words = [
            '露西亞', '露醬', 'ASMR', '撒嬌', '親親', '抱抱',
            '膝枕', '溫柔', '甜蜜', '陪伴', '一起', '幸福'
        ]
        
        logger.info("語義關鍵詞配置初始化完成")
    
    def get_keywords_by_category(self, category: str) -> List[str]:
        """根據類別獲取關鍵詞"""
        return self.semantic_keywords.get(category, [])
    
    def get_topic_keywords(self, topic: str) -> List[str]:
        """根據話題獲取關鍵詞"""
        return self.topic_mapping.get(topic, [])
    
    def get_emotion_weight(self, emotion: str) -> float:
        """獲取情感權重"""
        return self.emotion_weights.get(emotion, 0.0)
    
    def get_intimacy_factor(self, factor_type: str) -> float:
        """獲取親密度因子"""
        return self.intimacy_factors.get(factor_type, 0.0)
    
    def add_custom_word(self, word: str):
        """添加自定義詞彙"""
        if word not in self.custom_words:
            self.custom_words.append(word)
    
    def find_matching_keywords(self, text: str) -> Dict[str, List[str]]:
        """在文本中找到匹配的關鍵詞"""
        matches = {}
        
        for category, keywords in self.semantic_keywords.items():
            found_keywords = [keyword for keyword in keywords if keyword in text]
            if found_keywords:
                matches[category] = found_keywords
        
        return matches
    
    def get_all_categories(self) -> List[str]:
        """獲取所有關鍵詞類別"""
        return list(self.semantic_keywords.keys())

# 全局實例
keyword_config = SemanticKeywordConfig()
