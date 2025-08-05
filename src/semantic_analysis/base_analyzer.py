#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
語義分析基礎模組
提供所有語義分析模組的共同介面和基礎功能
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from .keyword_config import keyword_config

logger = logging.getLogger(__name__)

class BaseSemanticAnalyzer(ABC):
    """語義分析模組基類"""
    
    def __init__(self, chat_instance=None):
        """
        初始化語義分析模組
        
        Args:
            chat_instance: RushiaLoRAChat 實例
        """
        self.chat_instance = chat_instance
        self.module_name = self.__class__.__name__
        logger.info(f"初始化語義分析模組: {self.module_name}")
    
    @abstractmethod
    def analyze(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析用戶輸入
        
        Args:
            user_input: 用戶輸入文字
            context: 可選的上下文資訊
            
        Returns:
            Dict: 分析結果
        """
        pass
    
    def get_semantic_keywords(self) -> Dict[str, List[str]]:
        """獲取語義關鍵詞庫"""
        return keyword_config.semantic_keywords
    
    def get_keyword_config(self):
        """獲取關鍵詞配置實例"""
        return keyword_config
    
    def get_jieba_available(self) -> bool:
        """檢查jieba是否可用"""
        if self.chat_instance:
            return getattr(self.chat_instance, 'jieba_available', False)
        return False
    
    def extract_keywords(self, text: str, top_k: int = 8) -> List[str]:
        """
        提取關鍵詞
        
        Args:
            text: 輸入文字
            top_k: 返回關鍵詞數量
            
        Returns:
            List[str]: 關鍵詞列表
        """
        if self.get_jieba_available():
            try:
                import jieba.analyse
                return jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
            except ImportError:
                logger.warning("jieba不可用，使用基礎分詞")
        
        # 基礎分詞
        return text.split()[:top_k]
    
    def find_semantic_keywords(self, text: str) -> List[tuple]:
        """
        查找語義關鍵詞
        
        Args:
            text: 輸入文字
            
        Returns:
            List[tuple]: (類別, 關鍵詞) 的列表
        """
        semantic_keywords = self.get_semantic_keywords()
        found_keywords = []
        
        for category, words in semantic_keywords.items():
            for word in words:
                if word in text:
                    found_keywords.append((category, word))
        
        return found_keywords
    
    def is_keyword_match(self, text: str, keywords: List[str]) -> bool:
        """
        檢查文字是否包含指定關鍵詞
        
        Args:
            text: 輸入文字
            keywords: 關鍵詞列表
            
        Returns:
            bool: 是否匹配
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)
    
    def log_analysis(self, user_input: str, result: Dict[str, Any], analysis_type: str = "general"):
        """記錄分析結果（用於調試）"""
        logger.debug(f"[{self.module_name}] {analysis_type} - 輸入: {user_input[:30]}... 結果: {str(result)[:100]}...")
    
    def validate_result(self, result: Dict[str, Any]) -> bool:
        """驗證分析結果是否有效"""
        return isinstance(result, dict) and len(result) > 0
