#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
語義分析管理器
統一管理所有語義分析模組，提供簡潔的介面
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from .emotion_analyzer import EmotionAnalyzer
from .intent_recognizer import IntentRecognizer
from .intimacy_calculator import IntimacyCalculator
from .context_analyzer import ContextAnalyzer

logger = logging.getLogger(__name__)

class SemanticAnalysisManager:
    """語義分析管理器 - 統一所有語義分析功能"""
    
    def __init__(self, chat_instance=None):
        """
        初始化語義分析管理器
        
        Args:
            chat_instance: RushiaLoRAChat 實例
        """
        self.chat_instance = chat_instance
        
        # 初始化各個分析器
        self.emotion_analyzer = EmotionAnalyzer(chat_instance)
        self.intent_recognizer = IntentRecognizer(chat_instance)
        self.intimacy_calculator = IntimacyCalculator(chat_instance)
        self.context_analyzer = ContextAnalyzer(chat_instance)
        
        # 只在非啟動模式時記錄初始化訊息
        if logger.isEnabledFor(logging.INFO):
            logger.info("語義分析管理器初始化完成")
    
    def analyze_user_input(self, user_input: str, conversation_history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """
        完整分析用戶輸入
        
        Args:
            user_input: 用戶輸入文字
            conversation_history: 對話歷史（可選）
            
        Returns:
            Dict: 完整的語義分析結果
        """
        # 開始計時
        import time
        start_time = time.time()
        
        # 準備結果容器
        result = {
            'user_input': user_input,
            'analysis_timestamp': time.time(),
            'emotion_analysis': {},
            'intent_analysis': {},
            'intimacy_analysis': {},
            'context_analysis': {},
            'overall_summary': {},
            'processing_time': 0.0
        }
        
        try:
            # 1. 情感分析
            logger.debug("開始情感分析")
            result['emotion_analysis'] = self.emotion_analyzer.analyze(user_input)
            
            # 2. 意圖識別
            logger.debug("開始意圖識別")
            result['intent_analysis'] = self.intent_recognizer.analyze(user_input)
            
            # 3. 親密度計算
            logger.debug("開始親密度計算")
            result['intimacy_analysis'] = self.intimacy_calculator.analyze(user_input)
            
            # 4. 上下文分析（如果有對話歷史）
            if conversation_history:
                logger.debug("開始上下文分析")
                result['context_analysis'] = self.context_analyzer.analyze(conversation_history)
            
            # 5. 生成整體摘要
            result['overall_summary'] = self._generate_overall_summary(result)
            
            # 6. 添加兼容性映射 (為了與core.py兼容)
            result['intent_recognition'] = {
                'primary_intent': result['overall_summary'].get('primary_intent', 'unknown'),
                'conversation_intent': result['intent_analysis'].get('conversation_intent', 'casual'),
                'intent_type': result['intent_analysis'].get('intent_type', 'statement'),
                'confidence': result['intent_analysis'].get('confidence', 0.0)
            }
            
            # 計算處理時間
            result['processing_time'] = time.time() - start_time
            
            logger.debug(f"語義分析完成，耗時: {result['processing_time']:.3f}秒")
            
        except Exception as e:
            logger.error(f"語義分析過程中發生錯誤: {e}")
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
        
        return result
    
    def analyze_emotion_only(self, user_input: str) -> Dict[str, Any]:
        """僅進行情感分析"""
        return self.emotion_analyzer.analyze(user_input)
    
    def analyze_intent_only(self, user_input: str) -> Dict[str, Any]:
        """僅進行意圖識別"""
        return self.intent_recognizer.analyze(user_input)
    
    def analyze_intimacy_only(self, user_input: str) -> Dict[str, Any]:
        """僅進行親密度計算"""
        return self.intimacy_calculator.analyze(user_input)
    
    def analyze_context_only(self, conversation_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """僅進行上下文分析"""
        return self.context_analyzer.analyze(conversation_history)
    
    def get_legacy_intent_format(self, user_input: str) -> Dict[str, Any]:
        """
        獲取與原始系統兼容的意圖格式
        為了保持與現有系統的兼容性
        
        Args:
            user_input: 用戶輸入文字
            
        Returns:
            Dict: 原始格式的意圖分析結果
        """
        # 獲取新系統的分析結果
        emotion_result = self.emotion_analyzer.analyze(user_input)
        intent_result = self.intent_recognizer.analyze(user_input)
        intimacy_result = self.intimacy_calculator.analyze(user_input)
        
        # 轉換為原始格式
        legacy_intent = {
            'emotion': emotion_result['emotion'],
            'emotion_intensity': emotion_result['emotion_intensity'],
            'type': intent_result['intent_type'],
            'keywords': intent_result.get('question_words', []) + intent_result.get('action_words', []),
            'semantic_keywords': emotion_result['emotion_keywords'] + intimacy_result['intimacy_keywords'],
            'is_question': intent_result['is_question'],
            'response_expectation': intent_result.get('response_expectation', 'normal'),  # 🔥 新增：傳遞回應期望
            'is_about_action': len(intent_result.get('action_words', [])) > 0,
            'affection_level': intimacy_result['affection_level'],
            'intimacy_score': intimacy_result['intimacy_score'],
            'intimacy_keywords': [keyword for category, keyword in intimacy_result['intimacy_keywords']],
            'topic': intent_result['topic'],
            'response_expectation': intent_result['response_expectation'],
            'conversation_intent': intent_result['conversation_intent'],
            'time_sensitivity': intent_result['time_sensitivity']
        }
        
        return legacy_intent
    
    def _generate_overall_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成整體分析摘要"""
        summary = {
            'dominant_emotion': 'neutral',
            'primary_intent': 'casual',
            'intimacy_level': 'stranger',
            'response_strategy': 'normal',
            'priority_level': 'normal',
            'special_handling_required': False,
            'recommended_response_type': 'casual',
            'confidence_score': 0.0
        }
        
        emotion_analysis = analysis_result.get('emotion_analysis', {})
        intent_analysis = analysis_result.get('intent_analysis', {})
        intimacy_analysis = analysis_result.get('intimacy_analysis', {})
        context_analysis = analysis_result.get('context_analysis', {})
        
        # 主要情感
        summary['dominant_emotion'] = emotion_analysis.get('emotion', 'neutral')
        
        # 主要意圖 - 映射到標準化的意圖名稱
        conversation_intent = intent_analysis.get('conversation_intent', 'casual')
        intent_mapping = {
            'greeting': 'greeting',                    # 新增：問候語映射
            'seeking_comfort': 'emotional_support',
            'expressing_love': 'intimate_expression', 
            'asking_info': 'question_asking',
            'casual': 'casual_chat',
            'work_stress': 'emotional_support'
        }
        summary['primary_intent'] = intent_mapping.get(conversation_intent, 'casual_chat')
        
        # 親密度等級
        summary['intimacy_level'] = intimacy_analysis.get('intimacy_level', 'stranger')
        
        # 決定回應策略
        if intent_analysis.get('conversation_intent') == 'seeking_comfort':
            summary['response_strategy'] = 'supportive'
            summary['priority_level'] = 'high'
        elif intimacy_analysis.get('intimacy_score', 0) >= 2.0:
            summary['response_strategy'] = 'intimate'
            summary['priority_level'] = 'high'
        elif intent_analysis.get('urgency_level') == 'high':
            summary['response_strategy'] = 'urgent'
            summary['priority_level'] = 'high'
        elif emotion_analysis.get('emotion') == 'negative':
            summary['response_strategy'] = 'caring'
            summary['priority_level'] = 'medium'
        else:
            summary['response_strategy'] = 'normal'
            summary['priority_level'] = 'normal'
        
        # 特殊處理需求
        if (emotion_analysis.get('emotion_intensity', 0) < -0.5 or
            intimacy_analysis.get('intimacy_score', 0) >= 3.0 or
            intent_analysis.get('urgency_level') == 'high'):
            summary['special_handling_required'] = True
        
        # 推薦回應類型
        if intent_analysis.get('topic') == 'food':
            summary['recommended_response_type'] = 'food'
        elif intent_analysis.get('topic') == 'intimate':
            summary['recommended_response_type'] = 'intimate'
        elif intent_analysis.get('conversation_intent') == 'seeking_comfort':
            summary['recommended_response_type'] = 'emotional_support'
        elif intent_analysis.get('topic') == 'greeting':
            summary['recommended_response_type'] = 'greeting'
        elif intent_analysis.get('topic') == 'time_aware':
            summary['recommended_response_type'] = 'time_aware'
        else:
            summary['recommended_response_type'] = 'daily_chat'
        
        # 計算整體信心度
        confidences = []
        if emotion_analysis.get('confidence'):
            confidences.append(emotion_analysis['confidence'])
        if intent_analysis.get('confidence'):
            confidences.append(intent_analysis['confidence'])
        if intimacy_analysis.get('confidence'):
            confidences.append(intimacy_analysis['confidence'])
        if context_analysis.get('confidence'):
            confidences.append(context_analysis['confidence'])
        
        if confidences:
            summary['confidence_score'] = sum(confidences) / len(confidences)
        
        return summary
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """獲取分析統計資訊"""
        return {
            'emotion_analyzer_status': 'active',
            'intent_recognizer_status': 'active',
            'intimacy_calculator_status': 'active',
            'context_analyzer_status': 'active',
            'total_analyzers': 4,
            'initialization_time': getattr(self, '_init_time', 'unknown')
        }
    
    # 以下方法提供給主程式使用，保持與原始介面的兼容性
    
    def analyze_comprehensive(self, user_input: str, conversation_history: List[Tuple[str, str]] = None, 
                            user_profile: Dict[str, Any] = None, context_cache: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        進行全面的語義分析
        
        Args:
            user_input: 用戶輸入
            conversation_history: 對話歷史
            user_profile: 用戶檔案
            context_cache: 上下文緩存
            
        Returns:
            Dict: 包含所有分析結果的字典
        """
        result = {
            'emotion': {},
            'intent': {},
            'intimacy': {},
            'context': {},
            'overall': {}
        }
        
        try:
            # 情感分析
            emotion_result = self.emotion_analyzer.analyze(user_input)
            result['emotion'] = {
                'type': emotion_result.get('emotion', 'neutral'),
                'intensity': emotion_result.get('emotion_intensity', 0.0),
                'confidence': emotion_result.get('confidence', 0.0),
                'keywords': emotion_result.get('emotion_keywords', [])
            }
            
            # 意圖識別
            intent_result = self.intent_recognizer.analyze(user_input)
            result['intent'] = {
                'topic': intent_result.get('topic'),
                'type': intent_result.get('intent_type', 'statement'),
                'conversation_intent': intent_result.get('conversation_intent', 'casual'),
                'is_question': intent_result.get('is_question', False),
                'time_sensitivity': intent_result.get('time_sensitivity', False),
                'response_expectation': intent_result.get('response_expectation', 'normal'),
                'confidence': intent_result.get('confidence', 0.0)
            }
            
            # 親密度計算
            intimacy_result = self.intimacy_calculator.analyze(user_input)
            result['intimacy'] = {
                'score': intimacy_result.get('intimacy_score', 0.0),
                'level': intimacy_result.get('intimacy_level', 'stranger'),
                'affection_level': intimacy_result.get('affection_level', 0),
                'keywords': intimacy_result.get('intimacy_keywords', []),
                'confidence': intimacy_result.get('confidence', 0.0)
            }
            
            # 上下文分析
            if conversation_history:
                context_result = self.context_analyzer.analyze(conversation_history)
                result['context'] = context_result
            else:
                result['context'] = self._get_default_context()
            
            # 整體分析
            result['overall'] = self._generate_overall_summary({
                'emotion_analysis': emotion_result,
                'intent_analysis': intent_result,
                'intimacy_analysis': intimacy_result,
                'context_analysis': result['context']
            })
            
        except Exception as e:
            logger.error(f"綜合分析錯誤: {e}")
            result['error'] = str(e)
        
        return result
    
    def analyze_emotion(self, user_input: str) -> Dict[str, Any]:
        """分析情感（簡化介面）"""
        return self.emotion_analyzer.analyze(user_input)
    
    def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """分析意圖（簡化介面）"""
        return self.intent_recognizer.analyze(user_input)
    
    def analyze_intimacy(self, user_input: str) -> Dict[str, Any]:
        """分析親密度（簡化介面）"""
        return self.intimacy_calculator.analyze(user_input)
    
    def analyze_context(self, conversation_history: List[Tuple[str, str]], 
                       user_profile: Dict[str, Any] = None, 
                       context_cache: Dict[str, Any] = None) -> Dict[str, Any]:
        """分析上下文（簡化介面）"""
        return self.context_analyzer.analyze(conversation_history)
    
    def _get_default_context(self) -> Dict[str, Any]:
        """獲取默認上下文"""
        return {
            'recent_emotions': [],
            'emotion_trend': 'stable',
            'conversation_flow': [],
            'user_affection_expressed': False,
            'intimacy_level': 0,
            'intimacy_trend': 'stable',
            'topic_consistency': True,
            'topic_changes': 0,
            'response_length_trend': [],
            'conversation_depth': 0,
            'user_engagement': 'medium',
            'preferred_style': 'unknown',
            'conversation_duration': 0,
            'silence_detection': False,
            'confidence': 0.5
        }
