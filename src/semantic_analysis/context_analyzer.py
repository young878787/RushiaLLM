#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文分析模組
專門處理對話上下文的分析和理解
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .base_analyzer import BaseSemanticAnalyzer

logger = logging.getLogger(__name__)

class ContextAnalyzer(BaseSemanticAnalyzer):
    """上下文分析器"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 用戶參與度評估閾值
        self.engagement_thresholds = {
            'high': 30,    # 平均輸入長度
            'medium': 10,  # 平均輸入長度
            'low': 5       # 平均輸入長度
        }
        
        # 對話深度計算權重
        self.depth_weights = {
            'emotion_intensity': 2.0,
            'response_length': 0.02,  # 每50字符=1分
            'topic_consistency': 1.0,
            'intimacy_score': 1.5
        }
    
    def analyze(self, conversation_history: List[Tuple[str, str]], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析對話上下文
        
        Args:
            conversation_history: 對話歷史 [(用戶輸入, 機器人回應), ...]
            context: 可選的額外上下文資訊
            
        Returns:
            Dict: 上下文分析結果
        """
        result = {
            'recent_emotions': [],
            'emotion_trend': 'stable',  # improving, declining, stable
            'conversation_flow': [],
            'user_affection_expressed': False,
            'intimacy_level': 0.0,
            'intimacy_trend': 'stable',  # increasing, decreasing, stable
            'topic_consistency': True,
            'topic_changes': 0,
            'response_length_trend': [],
            'conversation_depth': 0.0,
            'user_engagement': 'medium',  # low, medium, high
            'preferred_style': 'unknown',  # casual, intimate, supportive
            'conversation_duration': 0,
            'silence_detection': False,
            'conversation_quality': 'normal',  # poor, normal, good, excellent
            'user_satisfaction_indicator': 'neutral',  # negative, neutral, positive
            'context_richness': 0.0,  # 上下文豐富度 0-1
            'confidence': 0.0
        }
        
        if not conversation_history:
            return result
        
        # 分析最近的對話（最多10輪）
        recent_history = conversation_history[-10:]
        
        # 1. 基礎統計分析
        result.update(self._analyze_basic_statistics(recent_history))
        
        # 2. 情感和親密度分析
        result.update(self._analyze_emotional_intimacy_context(recent_history))
        
        # 3. 話題一致性分析
        result.update(self._analyze_topic_consistency(recent_history))
        
        # 4. 用戶參與度分析
        result.update(self._analyze_user_engagement(recent_history))
        
        # 5. 對話品質評估
        result.update(self._assess_conversation_quality(recent_history))
        
        # 6. 用戶偏好風格分析
        result['preferred_style'] = self._analyze_preferred_style(recent_history)
        
        # 7. 計算對話深度
        result['conversation_depth'] = self._calculate_conversation_depth(result)
        
        # 8. 計算上下文豐富度
        result['context_richness'] = self._calculate_context_richness(result)
        
        # 9. 計算分析信心度
        result['confidence'] = self._calculate_confidence(result, len(recent_history))
        
        # 記錄分析結果
        self.log_analysis(f"{len(recent_history)}輪對話", result, "context_analysis")
        
        return result
    
    def _analyze_basic_statistics(self, conversation_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """分析基礎統計資料"""
        result = {
            'conversation_duration': len(conversation_history),
            'response_length_trend': []
        }
        
        # 分析回應長度趨勢
        response_lengths = [len(response) for _, response in conversation_history]
        result['response_length_trend'] = response_lengths
        
        return result
    
    def _analyze_emotional_intimacy_context(self, conversation_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """分析情感和親密度上下文"""
        result = {
            'recent_emotions': [],
            'emotion_trend': 'stable',
            'intimacy_level': 0.0,
            'intimacy_trend': 'stable',
            'user_affection_expressed': False
        }
        
        emotion_scores = []
        intimacy_scores = []
        
        # 需要導入其他分析器
        if self.chat_instance:
            for user_msg, bot_response in conversation_history:
                # 使用主程式的分析方法
                if hasattr(self.chat_instance, '_analyze_user_intent'):
                    intent = self.chat_instance._analyze_user_intent(user_msg)
                    
                    result['recent_emotions'].append(intent.get('emotion', 'neutral'))
                    emotion_scores.append(intent.get('emotion_intensity', 0.0))
                    intimacy_scores.append(intent.get('intimacy_score', 0.0))
                    
                    if intent.get('affection_level', 0) > 0:
                        result['user_affection_expressed'] = True
        
        # 分析情感趨勢
        if len(emotion_scores) >= 3:
            result['emotion_trend'] = self._calculate_trend(emotion_scores)
        
        # 分析親密度趨勢
        if len(intimacy_scores) >= 3:
            result['intimacy_trend'] = self._calculate_trend(intimacy_scores, threshold=0.5)
            result['intimacy_level'] = sum(intimacy_scores) / len(intimacy_scores)
        
        return result
    
    def _analyze_topic_consistency(self, conversation_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """分析話題一致性"""
        result = {
            'topic_consistency': True,
            'topic_changes': 0,
            'conversation_flow': []
        }
        
        topics = []
        
        if self.chat_instance and hasattr(self.chat_instance, '_analyze_user_intent'):
            for user_msg, _ in conversation_history:
                intent = self.chat_instance._analyze_user_intent(user_msg)
                topic = intent.get('topic')
                if topic:
                    topics.append(topic)
                    result['conversation_flow'].append(topic)
        
        # 計算話題變化
        unique_topics = set(topics)
        if len(topics) > 1:
            result['topic_changes'] = len(unique_topics) - 1
            result['topic_consistency'] = len(unique_topics) <= 2
        
        return result
    
    def _analyze_user_engagement(self, conversation_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """分析用戶參與度"""
        result = {'user_engagement': 'medium'}
        
        # 計算平均輸入長度
        user_inputs = [user_msg for user_msg, _ in conversation_history]
        if user_inputs:
            avg_input_length = sum(len(msg) for msg in user_inputs) / len(user_inputs)
            
            if avg_input_length >= self.engagement_thresholds['high']:
                result['user_engagement'] = 'high'
            elif avg_input_length <= self.engagement_thresholds['low']:
                result['user_engagement'] = 'low'
            else:
                result['user_engagement'] = 'medium'
        
        return result
    
    def _assess_conversation_quality(self, conversation_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """評估對話品質"""
        result = {
            'conversation_quality': 'normal',
            'user_satisfaction_indicator': 'neutral'
        }
        
        # 分析用戶回應模式
        user_inputs = [user_msg.lower() for user_msg, _ in conversation_history]
        
        # 正面指標
        positive_indicators = ['謝謝', '好', '開心', '喜歡', '愛', '棒', '不錯', '滿意']
        positive_count = sum(1 for msg in user_inputs for indicator in positive_indicators if indicator in msg)
        
        # 負面指標
        negative_indicators = ['不好', '無聊', '沒意思', '不喜歡', '煩', '差', '不滿']
        negative_count = sum(1 for msg in user_inputs for indicator in negative_indicators if indicator in msg)
        
        # 重複性指標（可能表示不滿意）
        repetitive_responses = ['嗯', '哦', '喔', 'ok', '好的']
        repetitive_count = sum(1 for msg in user_inputs if msg.strip() in repetitive_responses)
        
        # 評估對話品質
        total_messages = len(user_inputs)
        if total_messages > 0:
            positive_ratio = positive_count / total_messages
            negative_ratio = negative_count / total_messages
            repetitive_ratio = repetitive_count / total_messages
            
            if positive_ratio > 0.3:
                result['conversation_quality'] = 'excellent'
                result['user_satisfaction_indicator'] = 'positive'
            elif positive_ratio > 0.1 and negative_ratio < 0.1:
                result['conversation_quality'] = 'good'
                result['user_satisfaction_indicator'] = 'positive'
            elif negative_ratio > 0.2 or repetitive_ratio > 0.4:
                result['conversation_quality'] = 'poor'
                result['user_satisfaction_indicator'] = 'negative'
            else:
                result['conversation_quality'] = 'normal'
                result['user_satisfaction_indicator'] = 'neutral'
        
        return result
    
    def _analyze_preferred_style(self, conversation_history: List[Tuple[str, str]]) -> str:
        """分析用戶偏好的對話風格"""
        if not self.chat_instance or not hasattr(self.chat_instance, '_analyze_user_intent'):
            return 'unknown'
        
        intimacy_count = 0
        negative_emotion_count = 0
        question_count = 0
        total_messages = len(conversation_history)
        
        for user_msg, _ in conversation_history:
            intent = self.chat_instance._analyze_user_intent(user_msg)
            
            if intent.get('intimacy_score', 0) > 1.0:
                intimacy_count += 1
            
            if intent.get('emotion') == 'negative' and intent.get('emotion_intensity', 0) < -0.3:
                negative_emotion_count += 1
            
            if intent.get('is_question', False):
                question_count += 1
        
        if total_messages == 0:
            return 'unknown'
        
        # 計算比例
        intimacy_ratio = intimacy_count / total_messages
        negative_ratio = negative_emotion_count / total_messages
        question_ratio = question_count / total_messages
        
        # 判斷偏好風格
        if intimacy_ratio > 0.4:
            return 'intimate'
        elif negative_ratio > 0.3:
            return 'supportive'
        elif question_ratio > 0.4:
            return 'informative'
        else:
            return 'casual'
    
    def _calculate_conversation_depth(self, analysis_result: Dict[str, Any]) -> float:
        """計算對話深度"""
        depth = 0.0
        
        # 情感強度貢獻
        if analysis_result['recent_emotions']:
            emotion_variety = len(set(analysis_result['recent_emotions']))
            depth += emotion_variety * self.depth_weights['emotion_intensity']
        
        # 回應長度貢獻
        if analysis_result['response_length_trend']:
            avg_response_length = sum(analysis_result['response_length_trend']) / len(analysis_result['response_length_trend'])
            depth += avg_response_length * self.depth_weights['response_length']
        
        # 話題一致性貢獻（一致性高表示深入）
        if analysis_result['topic_consistency']:
            depth += self.depth_weights['topic_consistency']
        
        # 親密度貢獻
        depth += analysis_result['intimacy_level'] * self.depth_weights['intimacy_score']
        
        return min(depth, 10.0)  # 限制最大值
    
    def _calculate_context_richness(self, analysis_result: Dict[str, Any]) -> float:
        """計算上下文豐富度"""
        richness = 0.0
        
        # 對話長度貢獻
        richness += min(analysis_result['conversation_duration'] / 10, 0.3)
        
        # 情感多樣性貢獻
        if analysis_result['recent_emotions']:
            emotion_variety = len(set(analysis_result['recent_emotions']))
            richness += min(emotion_variety / 3, 0.2)
        
        # 話題多樣性貢獻
        if analysis_result['conversation_flow']:
            topic_variety = len(set(analysis_result['conversation_flow']))
            richness += min(topic_variety / 5, 0.2)
        
        # 用戶參與度貢獻
        engagement_scores = {'low': 0.1, 'medium': 0.15, 'high': 0.3}
        richness += engagement_scores.get(analysis_result['user_engagement'], 0.1)
        
        return min(richness, 1.0)
    
    def _calculate_trend(self, scores: List[float], threshold: float = 0.2) -> str:
        """計算數值趨勢"""
        if len(scores) < 3:
            return 'stable'
        
        mid_point = len(scores) // 2
        earlier_avg = sum(scores[:mid_point]) / mid_point
        later_avg = sum(scores[mid_point:]) / (len(scores) - mid_point)
        
        if later_avg > earlier_avg + threshold:
            return 'increasing'
        elif later_avg < earlier_avg - threshold:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_confidence(self, analysis_result: Dict[str, Any], conversation_length: int) -> float:
        """計算分析信心度"""
        confidence = 0.0
        
        # 基礎信心度（基於對話長度）
        confidence += min(conversation_length / 10, 0.4)
        
        # 有明確趨勢加分
        if analysis_result['emotion_trend'] != 'stable':
            confidence += 0.1
        
        if analysis_result['intimacy_trend'] != 'stable':
            confidence += 0.1
        
        # 高參與度加分
        if analysis_result['user_engagement'] == 'high':
            confidence += 0.2
        
        # 對話品質加分
        quality_scores = {'poor': 0.05, 'normal': 0.1, 'good': 0.15, 'excellent': 0.2}
        confidence += quality_scores.get(analysis_result['conversation_quality'], 0.1)
        
        return min(confidence, 1.0)
