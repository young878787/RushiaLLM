#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分析模組
專門處理用戶情感狀態的識別和分析
"""

import logging
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseSemanticAnalyzer

logger = logging.getLogger(__name__)

class EmotionAnalyzer(BaseSemanticAnalyzer):
    """情感分析器"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 情感權重配置
        self.emotion_weights = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
        
        # 情感強度閾值
        self.intensity_thresholds = {
            'very_weak': 0.2,
            'weak': 0.4,
            'medium': 0.6,
            'strong': 0.8,
            'very_strong': 1.0
        }
    
    def analyze(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析用戶輸入的情感狀態
        
        Args:
            user_input: 用戶輸入文字
            context: 可選的上下文資訊
            
        Returns:
            Dict: 情感分析結果
        """
        result = {
            'emotion': 'neutral',  # positive, negative, neutral
            'emotion_intensity': 0.0,  # 情感強度 -1.0 to 1.0
            'emotion_level': 'neutral',  # very_weak, weak, medium, strong, very_strong
            'positive_indicators': [],  # 正面情感指標
            'negative_indicators': [],  # 負面情感指標
            'emotion_keywords': [],  # 情感關鍵詞
            'confidence': 0.0  # 分析信心度
        }
        
        # 查找情感關鍵詞
        keyword_config = self.get_keyword_config()
        matches = keyword_config.find_matching_keywords(user_input)
        
        positive_score = 0
        negative_score = 0
        
        # 處理正面情感關鍵詞
        if 'emotion_positive' in matches:
            positive_score = len(matches['emotion_positive'])
            result['positive_indicators'].extend(matches['emotion_positive'])
            for word in matches['emotion_positive']:
                result['emotion_keywords'].append(('emotion_positive', word))
        
        # 處理負面情感關鍵詞  
        if 'emotion_negative' in matches:
            negative_score = len(matches['emotion_negative'])
            result['negative_indicators'].extend(matches['emotion_negative'])
            for word in matches['emotion_negative']:
                result['emotion_keywords'].append(('emotion_negative', word))
        
        # 特殊情況檢測
        positive_score += self._detect_special_positive_patterns(user_input)
        negative_score += self._detect_special_negative_patterns(user_input)
        
        # 計算情感強度和類型
        total_indicators = positive_score + negative_score
        
        if positive_score > negative_score:
            result['emotion'] = 'positive'
            result['emotion_intensity'] = min(positive_score / 3.0, 1.0)
        elif negative_score > positive_score:
            result['emotion'] = 'negative'
            result['emotion_intensity'] = -min(negative_score / 3.0, 1.0)
        else:
            result['emotion'] = 'neutral'
            result['emotion_intensity'] = 0.0
        
        # 確定情感強度等級
        abs_intensity = abs(result['emotion_intensity'])
        if abs_intensity >= self.intensity_thresholds['very_strong']:
            result['emotion_level'] = 'very_strong'
        elif abs_intensity >= self.intensity_thresholds['strong']:
            result['emotion_level'] = 'strong'
        elif abs_intensity >= self.intensity_thresholds['medium']:
            result['emotion_level'] = 'medium'
        elif abs_intensity >= self.intensity_thresholds['weak']:
            result['emotion_level'] = 'weak'
        elif abs_intensity >= self.intensity_thresholds['very_weak']:
            result['emotion_level'] = 'very_weak'
        else:
            result['emotion_level'] = 'neutral'
        
        # 計算信心度
        if total_indicators > 0:
            result['confidence'] = min(total_indicators / 5.0, 1.0)
        else:
            result['confidence'] = 0.1  # 基礎信心度
        
        # 記錄分析結果
        self.log_analysis(user_input, result, "emotion_analysis")
        
        return result
    
    def _detect_special_positive_patterns(self, text: str) -> float:
        """檢測特殊的正面情感模式"""
        positive_patterns = [
            # 語氣詞組合
            ['好', '呢'],
            ['真', '好'],
            ['超', '棒'],
            # 驚嘆模式
            ['哇'],
            ['嗯嗯'],
            # 表情符號模式
            ['♪', '♡', '～'],
            ['!', '！'],
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        # 檢測模式組合
        for pattern in positive_patterns:
            if all(word in text for word in pattern):
                score += 0.5
        
        # 檢測重複感嘆號
        if text.count('!') > 1 or text.count('！') > 1:
            score += 0.3
        
        # 檢測長度較長的積極回應
        if len(text) > 20 and any(word in text for word in ['很', '非常', '真的', '好']):
            score += 0.2
        
        return score
    
    def _detect_special_negative_patterns(self, text: str) -> float:
        """檢測特殊的負面情感模式"""
        negative_patterns = [
            # 工作壓力模式
            ['工作', '累'],
            ['忙', '累'], 
            ['壓力', '大'],
            # 孤獨模式
            ['一個人'],
            ['沒有人'],
            # 疲勞模式
            ['好累', '啊'],
            ['累', '死'],
            ['疲勞'],
            # 失望模式
            ['沒有', '意思'],
            ['好', '無聊'],
        ]
        
        score = 0.0
        text_lower = text.lower()
        
        # 檢測模式組合
        for pattern in negative_patterns:
            if all(word in text for word in pattern):
                score += 0.5
        
        # 檢測省略號（可能表示猶豫或疲憊）
        if '...' in text or '…' in text:
            score += 0.2
        
        # 檢測短促回應（可能表示情緒低落）
        if len(text.strip()) <= 3 and any(word in text for word in ['嗯', '哦', '喔']):
            score += 0.3
        
        return score
    
    def get_emotion_trend(self, recent_emotions: List[str]) -> str:
        """
        分析最近的情感趨勢
        
        Args:
            recent_emotions: 最近的情感列表
            
        Returns:
            str: 情感趨勢 (improving, declining, stable)
        """
        if len(recent_emotions) < 3:
            return 'stable'
        
        # 轉換情感為數值
        emotion_values = []
        for emotion in recent_emotions:
            if emotion == 'positive':
                emotion_values.append(1)
            elif emotion == 'negative':
                emotion_values.append(-1)
            else:
                emotion_values.append(0)
        
        # 比較前半段和後半段
        mid_point = len(emotion_values) // 2
        earlier_avg = sum(emotion_values[:mid_point]) / mid_point if mid_point > 0 else 0
        later_avg = sum(emotion_values[mid_point:]) / (len(emotion_values) - mid_point)
        
        if later_avg > earlier_avg + 0.3:
            return 'improving'
        elif later_avg < earlier_avg - 0.3:
            return 'declining'
        else:
            return 'stable'
    
    def is_emotional_support_needed(self, emotion_result: Dict[str, Any]) -> bool:
        """
        判斷是否需要情感支持
        
        Args:
            emotion_result: 情感分析結果
            
        Returns:
            bool: 是否需要情感支持
        """
        # 強烈負面情感需要支持
        if emotion_result['emotion'] == 'negative' and emotion_result['emotion_intensity'] < -0.4:
            return True
        
        # 包含特定負面關鍵詞
        negative_support_keywords = ['累', '疲勞', '難過', '傷心', '沮喪', '壓力', '煩惱', '孤單', '寂寞']
        if any(keyword in indicator for indicator in emotion_result['negative_indicators'] for keyword in negative_support_keywords):
            return True
        
        return False
