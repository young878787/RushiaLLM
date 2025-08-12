#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意圖識別模組
專門處理用戶意圖的識別和分類
"""

import logging
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseSemanticAnalyzer

logger = logging.getLogger(__name__)

class IntentRecognizer(BaseSemanticAnalyzer):
    """用戶意圖識別分析器"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 使用關鍵詞配置替代硬編碼
        self.keyword_config = self.get_keyword_config()
        
        # 對話意圖映射（部分使用關鍵詞配置）
        self.conversation_intents = {
            'casual': [],  # 默認
            'seeking_comfort': ['累', '難過', '傷心', '沮喪', '壓力', '寂寞', '孤單'],
            'expressing_love': self.keyword_config.get_keywords_by_category('intimacy_high'),
            'asking_info': self.keyword_config.get_keywords_by_category('question_words'),
            'work_stress': ['工作', '忙', '疲勞']
        }
        
        # 話題分類（使用關鍵詞配置）
        self.topic_categories = {
            'food': self.keyword_config.get_keywords_by_category('food_words'),
            'emotional_support': ['累', '疲勞', '難過', '傷心', '沮喪', '壓力', '煩惱', '寂寞', '孤單'],
            'intimate': (self.keyword_config.get_keywords_by_category('intimacy_high') + 
                        self.keyword_config.get_keywords_by_category('intimacy_medium')),
            'greeting': self.keyword_config.get_keywords_by_category('greeting_words'),
            'daily_chat': ['做什麼', '天氣', '遊戲', '工作', '學習', '今天', '明天'],
            'time_aware': self.keyword_config.get_keywords_by_category('time_words'),
            'companionship': self.keyword_config.get_keywords_by_category('companionship_words')
        }
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseSemanticAnalyzer

logger = logging.getLogger(__name__)

class IntentRecognizer(BaseSemanticAnalyzer):
    """意圖識別器"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 意圖類型定義
        self.intent_types = {
            'question': ['什麼', '哪裡', '怎麼', '為什麼', '誰', '何時', '如何', '可以嗎', '好嗎', '會不會', '能不能'],
            'request': ['想要', '希望', '可以', '能不能', '幫我', '麻煩'],
            'expression': ['愛', '喜歡', '討厭', '感謝', '謝謝'],
            'greeting': ['你好', '中午好', '早安',  'hello', 'hi', '午安', '早上好', '晚上好','下午好'],
            'statement': []  # 默認類型
        }
        
        # 對話意圖分類
        self.conversation_intents = {
            'casual': [],  # 默認
            'seeking_comfort': ['累', '難過', '傷心', '沮喪', '壓力', '寂寞', '孤單'],
            'expressing_love': ['愛你', '喜歡你', '想你', '親親', '抱抱'],
            'asking_info': ['什麼', '怎麼', '為什麼', '如何'],
            'work_stress': ['工作', '忙', '疲勞']
        }
        
        # 話題分類
        self.topic_categories = {
            'food': ['吃', '餓', '食物', '料理', '美食', '飯', '菜', '漢堡', '早餐', '午餐', '晚餐'],
            'emotional_support': ['累', '疲勞', '難過', '傷心', '沮喪', '壓力', '煩惱', '寂寞', '孤單'],
            'intimate': ['抱', '親', '愛', '喜歡', '想你', '陪伴', '一起', '牽手', '帶你', '陪你'],
            'greeting': ['你好', '中午好', '早安', 'hello', 'hi', '午安', '早上好', '晚上好', '下午好'],
            'daily_chat': ['做什麼', '天氣', '遊戲', '工作', '學習', '今天', '明天'],
            'time_aware': ['時間', '早上', '中午', '晚上', '睡覺', '起床', '現在'],
            'companionship': ['陪', '一起', '帶', '跟', '和', '同', '共同', '陪伴', '牽手', '牽著']
        }
    
    def analyze(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析用戶輸入的意圖
        
        Args:
            user_input: 用戶輸入文字
            context: 可選的上下文資訊
            
        Returns:
            Dict: 意圖分析結果
        """
        result = {
            'intent_type': 'statement',  # question, statement, request, expression, greeting
            'conversation_intent': 'casual',  # casual, seeking_comfort, expressing_love, asking_info, work_stress
            'topic': None,  # 主要話題
            'secondary_topics': [],  # 次要話題
            'is_question': False,
            'is_request': False,
            'is_expression': False,
            'is_greeting': False,
            'action_words': [],  # 動作相關詞彙
            'question_words': [],  # 問句詞彙
            'time_sensitivity': False,  # 是否有時間敏感性
            'response_expectation': 'normal',  # short, normal, detailed
            'urgency_level': 'normal',  # low, normal, high
            'confidence': 0.0
        }
        
        user_lower = user_input.lower()
        
        # 1. 識別基本意圖類型
        result.update(self._identify_intent_type(user_input))
        
        # 2. 識別對話意圖
        result['conversation_intent'] = self._identify_conversation_intent(user_input)
        
        # 3. 識別話題
        topics = self._identify_topics(user_input)
        if topics:
            result['topic'] = topics[0]  # 主要話題
            result['secondary_topics'] = topics[1:]  # 次要話題
        
        # 4. 檢測特殊組合
        result.update(self._detect_special_combinations(user_input))
        
        # 5. 分析時間敏感性
        result['time_sensitivity'] = self._detect_time_sensitivity(user_input)
        
        # 6. 確定回應期望
        result['response_expectation'] = self._determine_response_expectation(result)
        
        # 7. 評估緊急程度
        result['urgency_level'] = self._assess_urgency(user_input, result)
        
        # 8. 計算信心度
        result['confidence'] = self._calculate_confidence(result)
        
        # 記錄分析結果
        self.log_analysis(user_input, result, "intent_recognition")
        
        return result
    
    def _identify_intent_type(self, user_input: str) -> Dict[str, Any]:
        """識別基本意圖類型"""
        result = {
            'intent_type': 'statement',
            'is_question': False,
            'is_request': False,
            'is_expression': False,
            'is_greeting': False,
            'question_words': [],
            'action_words': []
        }
        
        user_lower = user_input.lower()
        
        # 檢測問句
        for word in self.intent_types['question']:
            if word in user_input:
                result['question_words'].append(word)
        
        if result['question_words'] or user_input.endswith('?') or user_input.endswith('？'):
            result['intent_type'] = 'question'
            result['is_question'] = True
        
        # 檢測請求
        elif any(word in user_input for word in self.intent_types['request']):
            result['intent_type'] = 'request'
            result['is_request'] = True
        
        # 檢測表達
        elif any(word in user_input for word in self.intent_types['expression']):
            result['intent_type'] = 'expression'
            result['is_expression'] = True
        
        # 檢測問候
        elif any(word in user_input for word in self.intent_types['greeting']):
            result['intent_type'] = 'greeting'
            result['is_greeting'] = True
        
        # 檢測動作詞
        action_words = ['做', '在做', '準備', '想要', '希望', '打算', '計劃', '開始', '結束', '進行', '帶', '去', '走', '來']
        for word in action_words:
            if word in user_input:
                result['action_words'].append(word)
        
        return result
    
    def _identify_conversation_intent(self, user_input: str) -> str:
        """識別對話意圖"""
        # 優先級排序檢測
        
        # 1. 問候語 (最高優先級)
        for word in self.intent_types['greeting']:
            if word in user_input:
                return 'greeting'
        
        # 2. 工作壓力 (特殊組合)
        if (any(word in user_input for word in ['工作', '忙']) and 
            any(word in user_input for word in ['累', '疲勞', '寂寞', '孤單'])):
            return 'work_stress'
        
        # 3. 尋求安慰
        comfort_indicators = 0
        for word in self.conversation_intents['seeking_comfort']:
            if word in user_input:
                comfort_indicators += 1
        
        if comfort_indicators >= 1:
            return 'seeking_comfort'
        
        # 4. 表達愛意
        for word in self.conversation_intents['expressing_love']:
            if word in user_input:
                return 'expressing_love'
        
        # 5. 詢問資訊
        for word in self.conversation_intents['asking_info']:
            if word in user_input:
                return 'asking_info'
        
        # 6. 默認為日常聊天
        return 'casual'
    
    def _identify_topics(self, user_input: str) -> List[str]:
        """識別話題（可能多個）"""
        identified_topics = []
        
        for topic, keywords in self.topic_categories.items():
            if any(keyword in user_input for keyword in keywords):
                identified_topics.append(topic)
        
        # 特殊組合檢測
        if (any(word in user_input for word in ['帶', '去', '走', '來']) and 
            any(word in user_input for word in ['吃', '漢堡', '食物'])):
            identified_topics.append('companionship_food')
        
        return identified_topics
    
    def _detect_special_combinations(self, user_input: str) -> Dict[str, Any]:
        """檢測特殊語義組合"""
        result = {}
        
        # 陪伴+食物組合
        if (any(word in user_input for word in ['帶', '去', '走', '來', '牽手', '陪']) and 
            any(word in user_input for word in ['吃', '漢堡', '食物'])):
            result['special_combination'] = 'companionship_food'
        
        # 親密+行動組合
        elif any(word in user_input for word in ['牽手', '帶你', '陪你', '一起']):
            result['special_combination'] = 'intimate_action'
        
        return result
    
    def _detect_time_sensitivity(self, user_input: str) -> bool:
        """檢測時間敏感性"""
        time_sensitive_words = [
            '現在', '馬上', '立刻', '趕快', '緊急', '急', '快',
            '今天', '明天', '昨天', '早上', '中午', '下午', '晚上', '睡覺', '起床',
            # 添加時間問候語
            '早安', '午安', '早上好', '中午好', '下午好', '晚上好',
            '時間', '現在幾點', '現在是', '什麼時候', '幾點',
            'おはよう', 'おやすみ', 'morning', 'afternoon', 'evening', 'night'
        ]
        
        return any(word in user_input for word in time_sensitive_words)
    
    def _determine_response_expectation(self, intent_result: Dict[str, Any]) -> str:
        """確定回應期望長度"""
        if intent_result['is_question']:
            return 'detailed'
        elif intent_result['conversation_intent'] == 'seeking_comfort':
            return 'detailed'
        elif intent_result['is_greeting']:
            return 'normal'
        elif intent_result.get('special_combination'):
            return 'detailed'
        else:
            return 'normal'
    
    def _assess_urgency(self, user_input: str, intent_result: Dict[str, Any]) -> str:
        """評估緊急程度"""
        high_urgency_indicators = ['急', '緊急', '馬上', '立刻', '趕快', '快']
        
        if any(word in user_input for word in high_urgency_indicators):
            return 'high'
        elif intent_result['conversation_intent'] == 'seeking_comfort':
            return 'high'
        elif intent_result['time_sensitivity']:
            return 'normal'
        else:
            return 'low'
    
    def _calculate_confidence(self, intent_result: Dict[str, Any]) -> float:
        """計算識別信心度"""
        confidence = 0.0
        
        # 基礎信心度
        confidence += 0.3
        
        # 明確的意圖類型加分
        if intent_result['is_question'] or intent_result['is_greeting']:
            confidence += 0.3
        
        # 有明確話題加分
        if intent_result['topic']:
            confidence += 0.2
        
        # 有特殊組合加分
        if intent_result.get('special_combination'):
            confidence += 0.2
        
        # 多個指標加分
        if len(intent_result['question_words']) > 1 or len(intent_result['action_words']) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
