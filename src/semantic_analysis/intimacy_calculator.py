#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
親密度計算模組
專門處理用戶與露西亞之間親密度的計算和分析
"""

import logging
from typing import Dict, List, Any, Optional
from .base_analyzer import BaseSemanticAnalyzer

logger = logging.getLogger(__name__)

class IntimacyCalculator(BaseSemanticAnalyzer):
    """親密度計算器"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 使用關鍵詞配置
        self.keyword_config = self.get_keyword_config()
        
        # 親密度計算因子（從配置獲取）
        self.intimacy_factors = self.keyword_config.intimacy_factors
        
        # 親密度關鍵詞分類（結合配置和自定義）
        self.intimacy_keywords = {
            'physical_contact': [
                '抱抱', '擁抱', '親親', '親吻', '摸頭', '撫摸', '膝枕', '靠著', '依偎',
                '緊緊抱', '用力抱', '抱緊', '貼著', '緊貼', '摟著', '輕撫'
            ],
            'emotional_expression': (
                self.keyword_config.get_keywords_by_category('intimacy_high') +
                self.keyword_config.get_keywords_by_category('intimacy_medium')
            ),
            'companionship': (
                self.keyword_config.get_keywords_by_category('companionship_words') +
                ['陪伴', '一起', '陪你', '帶你', '牽手', '陪在身邊', '不離開',
                 '永遠在一起', '共度', '相伴', '同行', '並肩']
            ),
            'romantic_expression': [
                '戀人', '男女朋友', '情侶', '約會', '浪漫', '告白', '求婚',
                '我的', '屬於我', '只屬於', '專屬', '獨有'
            ],
            'protective_care': [
                '保護你', '照顧你', '守護', '關心', '擔心', '心疼', '疼愛',
                '呵護', '愛護', '寵愛', '關懷', '體貼'
            ],
            'general_affection': [
                '可愛', '美麗', '好看', '喜歡', '不錯', '很棒', '滿意',
                '開心', '快樂', '舒服', '放鬆', '輕鬆'
            ]
        }
        
        # 親密度等級定義
        self.intimacy_levels = {
            0: 'stranger',      # 陌生人
            1: 'acquaintance',  # 熟人
            2: 'friend',        # 朋友
            3: 'close_friend',  # 親近朋友
            4: 'intimate',      # 親密
            5: 'very_intimate'  # 非常親密
        }
    
    def analyze(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析用戶輸入的親密度指標
        
        Args:
            user_input: 用戶輸入文字
            context: 可選的上下文資訊
            
        Returns:
            Dict: 親密度分析結果
        """
        result = {
            'intimacy_score': 0.0,  # 總親密度評分
            'affection_level': 0,   # 喜愛程度 (0-5)
            'intimacy_level': 'stranger',  # 親密度等級
            'intimacy_keywords': [],  # 找到的親密度關鍵詞
            'intimacy_categories': {},  # 各類別的評分
            'intimacy_trend': 'stable',  # 親密度趨勢
            'special_indicators': [],  # 特殊親密度指標
            'confidence': 0.0  # 分析信心度
        }
        
        # 1. 基礎親密度關鍵詞分析
        result.update(self._analyze_intimacy_keywords(user_input))
        
        # 2. 特殊親密度模式檢測
        result.update(self._detect_special_intimacy_patterns(user_input))
        
        # 3. 上下文親密度分析
        if context:
            result.update(self._analyze_contextual_intimacy(context))
        
        # 4. 計算最終親密度評分
        result['intimacy_score'] = self._calculate_final_score(result)
        
        # 5. 確定親密度等級
        result['affection_level'] = min(int(result['intimacy_score']), 5)
        result['intimacy_level'] = self.intimacy_levels.get(result['affection_level'], 'stranger')
        
        # 6. 計算信心度
        result['confidence'] = self._calculate_confidence(result)
        
        # 記錄分析結果
        self.log_analysis(user_input, result, "intimacy_analysis")
        
        return result
    
    def _analyze_intimacy_keywords(self, user_input: str) -> Dict[str, Any]:
        """分析親密度關鍵詞"""
        result = {
            'intimacy_keywords': [],
            'intimacy_categories': {}
        }
        
        total_score = 0.0
        
        for category, keywords in self.intimacy_keywords.items():
            category_score = 0.0
            found_keywords = []
            
            for keyword in keywords:
                if keyword in user_input:
                    found_keywords.append(keyword)
                    category_score += self.intimacy_factors.get(category, 0.5)
                    result['intimacy_keywords'].append((category, keyword))
            
            if found_keywords:
                result['intimacy_categories'][category] = {
                    'score': category_score,
                    'keywords': found_keywords,
                    'count': len(found_keywords)
                }
                total_score += category_score
        
        result['base_intimacy_score'] = total_score
        
        return result
    
    def _detect_special_intimacy_patterns(self, user_input: str) -> Dict[str, Any]:
        """檢測特殊親密度模式"""
        result = {'special_indicators': []}
        bonus_score = 0.0
        
        # 1. 多重親密表達
        intimate_expressions = ['愛', '喜歡', '想', '親', '抱']
        expression_count = sum(1 for expr in intimate_expressions if expr in user_input)
        if expression_count >= 2:
            result['special_indicators'].append('multiple_expressions')
            bonus_score += 0.5 * expression_count
        
        # 2. 強化詞組合
        intensifiers = ['很', '非常', '特別', '超級', '最', '好', '真的']
        intimate_words = ['愛', '喜歡', '想你', '可愛', '溫柔']
        
        for intensifier in intensifiers:
            for intimate_word in intimate_words:
                if intensifier in user_input and intimate_word in user_input:
                    result['special_indicators'].append(f'intensified_{intimate_word}')
                    bonus_score += 0.3
        
        # 3. 所有格表達 (表示佔有慾/親密關係)
        possessive_patterns = ['我的', '只有我', '屬於我', '專屬']
        for pattern in possessive_patterns:
            if pattern in user_input:
                result['special_indicators'].append('possessive_expression')
                bonus_score += 1.0
                break
        
        # 4. 永恆承諾表達
        eternal_patterns = ['永遠', '一直', '永不', '一輩子', 'forever']
        intimate_context = any(word in user_input for word in ['愛', '陪', '一起', '守護'])
        
        if any(pattern in user_input for pattern in eternal_patterns) and intimate_context:
            result['special_indicators'].append('eternal_commitment')
            bonus_score += 1.5
        
        # 5. 身體親密度表達
        physical_intimacy = ['緊緊', '輕輕', '慢慢', '溫柔地']
        physical_actions = ['抱', '摸', '親', '撫摸', '牽手']
        
        physical_score = 0
        for modifier in physical_intimacy:
            for action in physical_actions:
                if modifier in user_input and action in user_input:
                    physical_score += 0.4
        
        if physical_score > 0:
            result['special_indicators'].append('enhanced_physical_intimacy')
            bonus_score += physical_score
        
        # 6. 情感脆弱性表達 (表示信任)
        vulnerability_patterns = ['只對你', '不告訴別人', '祕密', '心事', '真心話']
        if any(pattern in user_input for pattern in vulnerability_patterns):
            result['special_indicators'].append('emotional_vulnerability')
            bonus_score += 0.8
        
        result['special_intimacy_bonus'] = bonus_score
        
        return result
    
    def _analyze_contextual_intimacy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析上下文中的親密度"""
        result = {}
        
        # 從上下文獲取歷史親密度
        if 'intimacy_level' in context:
            result['historical_intimacy'] = context['intimacy_level']
        
        # 分析親密度趨勢
        if 'recent_intimacy_scores' in context:
            scores = context['recent_intimacy_scores']
            if len(scores) >= 3:
                recent_avg = sum(scores[-2:]) / 2
                earlier_avg = sum(scores[:-2]) / len(scores[:-2])
                
                if recent_avg > earlier_avg + 0.5:
                    result['intimacy_trend'] = 'increasing'
                elif recent_avg < earlier_avg - 0.5:
                    result['intimacy_trend'] = 'decreasing'
                else:
                    result['intimacy_trend'] = 'stable'
        
        return result
    
    def _calculate_final_score(self, analysis_result: Dict[str, Any]) -> float:
        """計算最終親密度評分"""
        base_score = analysis_result.get('base_intimacy_score', 0.0)
        special_bonus = analysis_result.get('special_intimacy_bonus', 0.0)
        
        # 基礎分數 + 特殊獎勵
        final_score = base_score + special_bonus
        
        # 上下文調整
        if analysis_result.get('intimacy_trend') == 'increasing':
            final_score *= 1.1
        elif analysis_result.get('intimacy_trend') == 'decreasing':
            final_score *= 0.9
        
        # 限制最大值
        return min(final_score, 5.0)
    
    def _calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """計算分析信心度"""
        confidence = 0.0
        
        # 基礎信心度
        confidence += 0.2
        
        # 有親密度關鍵詞加分
        if analysis_result['intimacy_keywords']:
            confidence += min(len(analysis_result['intimacy_keywords']) * 0.1, 0.4)
        
        # 有特殊指標加分
        if analysis_result['special_indicators']:
            confidence += min(len(analysis_result['special_indicators']) * 0.15, 0.3)
        
        # 多個類別加分
        if len(analysis_result['intimacy_categories']) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def calculate_intimacy_trend(self, recent_scores: List[float]) -> str:
        """
        計算親密度趨勢
        
        Args:
            recent_scores: 最近的親密度評分列表
            
        Returns:
            str: 趨勢 (increasing, decreasing, stable)
        """
        if len(recent_scores) < 3:
            return 'stable'
        
        # 比較前半段和後半段
        mid_point = len(recent_scores) // 2
        earlier_avg = sum(recent_scores[:mid_point]) / mid_point
        later_avg = sum(recent_scores[mid_point:]) / (len(recent_scores) - mid_point)
        
        threshold = 0.5
        if later_avg > earlier_avg + threshold:
            return 'increasing'
        elif later_avg < earlier_avg - threshold:
            return 'decreasing'
        else:
            return 'stable'
    
    def is_intimate_context(self, intimacy_result: Dict[str, Any]) -> bool:
        """
        判斷是否為親密情境
        
        Args:
            intimacy_result: 親密度分析結果
            
        Returns:
            bool: 是否為親密情境
        """
        # 高親密度評分
        if intimacy_result['intimacy_score'] >= 2.0:
            return True
        
        # 包含身體接觸或浪漫表達
        high_intimacy_categories = ['physical_contact', 'romantic_expression']
        for category in high_intimacy_categories:
            if category in intimacy_result['intimacy_categories']:
                return True
        
        # 有特殊親密度指標
        intimate_indicators = ['multiple_expressions', 'possessive_expression', 'eternal_commitment']
        if any(indicator in intimacy_result['special_indicators'] for indicator in intimate_indicators):
            return True
        
        return False
