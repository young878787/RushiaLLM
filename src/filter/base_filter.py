#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基礎回應過濾器
定義所有過濾器的通用接口和基礎功能
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseResponseFilter(ABC):
    """回應過濾器基礎類"""
    
    def __init__(self, chat_instance=None):
        self.chat_instance = chat_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 統計信息
        self.stats = {
            'processed_count': 0,
            'modified_count': 0,
            'error_count': 0
        }
    
    @abstractmethod
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        過濾器主要方法
        
        Args:
            response: 原始回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            str: 處理後的回應
        """
        pass
    
    @abstractmethod
    def get_filter_name(self) -> str:
        """回傳過濾器名稱"""
        pass
    
    @abstractmethod
    def get_filter_description(self) -> str:
        """回傳過濾器描述"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取過濾器統計信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置統計信息"""
        self.stats = {
            'processed_count': 0,
            'modified_count': 0,
            'error_count': 0
        }
    
    def _should_filter(self, response: str, user_input: str = "", context: Dict = None) -> bool:
        """檢查是否需要進行過濾處理"""
        return bool(response and response.strip())
    
    def _apply_filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """應用過濾邏輯 - 子類可覆寫此方法"""
        return response
    
    def _get_debug_info(self, original: str, filtered: str) -> Dict:
        """取得調試資訊 - 子類可覆寫此方法"""
        return {
            'original_length': len(original),
            'filtered_length': len(filtered),
            'modified': original != filtered
        }
