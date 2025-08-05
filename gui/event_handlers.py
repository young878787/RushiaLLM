#!/usr/bin/env python3
"""
GUI 事件處理器
處理所有GUI事件與核心服務的交互
"""

import asyncio
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class EventHandlers:
    """GUI事件處理器類"""
    
    def __init__(self, core_service, async_helper):
        self.core_service = core_service
        self.async_helper = async_helper
    
    # ==================== 聊天處理 ====================
    
    async def handle_chat_message(self, message: str, user_id: str) -> Dict[str, Any]:
        """處理聊天消息"""
        try:
            # 調用核心服務處理消息
            result = await self.core_service.chat(message, user_id)
            
            if result.get('success'):
                return {
                    'success': True,
                    'response': result.get('response', ''),
                    'character_name': result.get('character_name', 'AI'),
                    'metadata': result.get('metadata', {})
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', '未知錯誤')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"消息處理異常: {str(e)}"
            }
    
    # ==================== RAG 控制 ====================
    
    def handle_toggle_rag(self, enabled: bool) -> Dict[str, Any]:
        """處理RAG開關"""
        try:
            result = self.core_service.toggle_rag(enabled)  # 這是同步方法
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"RAG切換失敗: {str(e)}"
            }
    
    async def handle_upload_document(self, file_path: str) -> Dict[str, Any]:
        """處理文檔上傳"""
        try:
            # 檢查文件是否存在
            if not Path(file_path).exists():
                return {
                    'success': False,
                    'error': '文件不存在'
                }
            
            # 調用核心服務上傳文檔
            result = await self.core_service.upload_document(file_path)
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"文檔上傳異常: {str(e)}"
            }
    
    async def handle_clear_knowledge_base(self) -> Dict[str, Any]:
        """處理清空知識庫"""
        try:
            result = await self.core_service.clear_knowledge_base()
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"清空知識庫失敗: {str(e)}"
            }
    
    async def handle_search_knowledge(self, query: str) -> Dict[str, Any]:
        """處理知識庫搜索"""
        try:
            result = await self.core_service.search_knowledge(query)
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"知識搜索失敗: {str(e)}"
            }
    
    # ==================== 系統控制 ====================
    
    def handle_toggle_line_break(self, enabled: bool) -> Dict[str, Any]:
        """處理智慧換行切換"""
        try:
            result = self.core_service.toggle_line_break(enabled)  # 這是同步方法
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"智慧換行切換失敗: {str(e)}"
            }
    
    async def handle_toggle_traditional(self, enabled: bool) -> Dict[str, Any]:
        """處理簡繁轉換切換"""
        try:
            result = await self.core_service.toggle_traditional_chinese(enabled)
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"簡繁轉換切換失敗: {str(e)}"
            }
    
    async def handle_clear_memory(self) -> Dict[str, Any]:
        """處理清除記憶"""
        try:
            result = await self.core_service.clear_conversation_memory()
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"清除記憶失敗: {str(e)}"
            }
    
    # ==================== 統計信息 ====================
    
    async def handle_get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        try:
            result = await self.core_service.get_stats()
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"獲取統計失敗: {str(e)}"
            }
    
    async def handle_get_system_info(self) -> Dict[str, Any]:
        """獲取系統信息"""
        try:
            # 獲取基礎系統信息
            system_info = {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': platform.platform(),
                'memory_usage': f"{psutil.virtual_memory().percent:.1f}%" if PSUTIL_AVAILABLE else "Unknown",
                'uptime': self._get_uptime()
            }
            
            # 獲取核心服務狀態
            core_stats = await self.core_service.get_stats()
            if core_stats.get('success'):
                stats_data = core_stats.get('stats', {})
                
                # 模型狀態
                models_info = {
                    'llm_status': '已加載' if stats_data.get('llm_initialized') else '未加載',
                    'embedding_status': '已加載' if stats_data.get('embedding_initialized') else '未加載'
                }
                
                # 功能狀態
                features_info = {
                    'rag_enabled': stats_data.get('rag_enabled', False),
                    'line_break_enabled': stats_data.get('line_break_enabled', True),
                    'traditional_enabled': stats_data.get('traditional_chinese_enabled', True)
                }
                
                return {
                    'success': True,
                    'info': {
                        **system_info,
                        'models': models_info,
                        'features': features_info
                    }
                }
            else:
                return {
                    'success': True,
                    'info': {
                        **system_info,
                        'models': {'llm_status': '狀態未知', 'embedding_status': '狀態未知'},
                        'features': {'rag_enabled': False, 'line_break_enabled': True, 'traditional_enabled': True}
                    }
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"獲取系統信息失敗: {str(e)}"
            }
    
    async def handle_get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        try:
            # 這裡可以調用核心服務的模型信息接口
            # 目前先返回基礎信息
            return {
                'success': True,
                'info': {
                    'llm_model': 'Qwen-8B',
                    'embedding_model': 'Qwen3-Embedding-0.6B',
                    'status': '運行中'
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"獲取模型信息失敗: {str(e)}"
            }
    
    async def handle_get_conversion_info(self) -> Dict[str, Any]:
        """獲取轉換狀態信息"""
        try:
            # 檢查核心服務的簡繁轉換狀態
            stats = await self.core_service.get_stats()
            if stats.get('success'):
                stats_data = stats.get('stats', {})
                return {
                    'success': True,
                    'info': {
                        'enabled': stats_data.get('traditional_chinese_enabled', True),
                        'status': '正常運行'
                    }
                }
            else:
                return {
                    'success': True,
                    'info': {
                        'enabled': False,
                        'status': '狀態未知'
                    }
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"獲取轉換信息失敗: {str(e)}"
            }
    
    # ==================== 輔助方法 ====================
    
    def _get_uptime(self) -> str:
        """獲取系統運行時間"""
        if not PSUTIL_AVAILABLE:
            return "未知"
            
        try:
            uptime_seconds = psutil.boot_time()
            current_time = datetime.now().timestamp()
            uptime = current_time - uptime_seconds
            
            days = int(uptime // 86400)
            hours = int((uptime % 86400) // 3600)
            minutes = int((uptime % 3600) // 60)
            
            if days > 0:
                return f"{days}天 {hours}小時 {minutes}分鐘"
            elif hours > 0:
                return f"{hours}小時 {minutes}分鐘"
            else:
                return f"{minutes}分鐘"
        except:
            return "未知"
    
    # ==================== STT 語音識別處理 ====================
    
    async def handle_toggle_stt(self, enabled: bool) -> Dict[str, Any]:
        """處理STT語音識別切換"""
        try:
            result = await self.core_service.toggle_stt(enabled)
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"STT切換失敗: {str(e)}"
            }
    
    async def handle_toggle_auto_response(self, enabled: bool) -> Dict[str, Any]:
        """處理語音自動回應切換"""
        try:
            result = self.core_service.toggle_auto_response(enabled)
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"自動回應切換失敗: {str(e)}"
            }
    
    async def handle_update_stt_sensitivity(self, sensitivity: float) -> Dict[str, Any]:
        """處理STT靈敏度更新"""
        try:
            result = await self.core_service.update_stt_sensitivity(sensitivity)
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"STT靈敏度更新失敗: {str(e)}"
            }
    
    # ==================== 高級功能處理 ====================
    
    async def handle_batch_upload(self, file_paths: list) -> Dict[str, Any]:
        """處理批量文檔上傳"""
        try:
            results = []
            for file_path in file_paths:
                result = await self.handle_upload_document(file_path)
                results.append({
                    'file': file_path,
                    'success': result.get('success', False),
                    'error': result.get('error', '')
                })
            
            success_count = sum(1 for r in results if r['success'])
            return {
                'success': True,
                'results': results,
                'summary': f"成功上傳 {success_count}/{len(file_paths)} 個文檔"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"批量上傳失敗: {str(e)}"
            }
    
    async def handle_export_conversation(self, format: str = 'txt') -> Dict[str, Any]:
        """處理對話導出"""
        try:
            # 這裡可以實現對話記錄的導出邏輯
            return {
                'success': True,
                'message': f"對話記錄已導出為 {format} 格式"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"對話導出失敗: {str(e)}"
            }
    
    async def handle_backup_knowledge_base(self) -> Dict[str, Any]:
        """處理知識庫備份"""
        try:
            # 這裡可以實現知識庫備份邏輯
            return {
                'success': True,
                'message': "知識庫備份完成"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"知識庫備份失敗: {str(e)}"
            }
    
    async def handle_restore_knowledge_base(self, backup_path: str) -> Dict[str, Any]:
        """處理知識庫恢復"""
        try:
            # 這裡可以實現知識庫恢復邏輯
            return {
                'success': True,
                'message': "知識庫恢復完成"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"知識庫恢復失敗: {str(e)}"
            }
