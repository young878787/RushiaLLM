#!/usr/bin/env python3
"""
VTuber AI 核心服務層
提供統一的AI服務接口，供不同前端調用
"""
import asyncio
import logging
import sys
import random
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator
from datetime import datetime
from pathlib import Path

from .llm_manager import LLMManager
from .rag_system import RAGSystem
from .STT import RealtimeSTTService, create_stt_service, TranscriptionResult
from .utils.logger import setup_logger
from .utils.system_optimizer import WindowsOptimizer
from .filter.smart_line_break_filter import SmartLineBreakFilter


class VTuberCoreService:
    """VTuber AI 核心服務類 - 封裝所有AI邏輯"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # AI 核心組件
        self.llm_manager: Optional[LLMManager] = None
        self.rag_system: Optional[RAGSystem] = None
        self.smart_line_break_filter: Optional[SmartLineBreakFilter] = None
        self.stt_service: Optional[RealtimeSTTService] = None
        self.rag_enabled = True
        
        # GUI 回調機制
        self.gui_voice_preview_callback = None
        self.gui_voice_status_callback = None
        
        # 過濾器控制
        self.line_break_enabled = config.get('vtuber', {}).get('response', {}).get('enable_line_break', True)
        
        # STT 控制
        self.stt_enabled = config.get('stt', {}).get('enabled', False)
        self.auto_response_enabled = config.get('stt', {}).get('auto_response', False)  # 是否自動回應語音輸入
        
        # 人性化對話節奏控制
        response_config = config.get('vtuber', {}).get('response', {})
        self.typing_simulation_enabled = response_config.get('enable_typing_simulation', True)
        self.typing_speed = response_config.get('typing_speed', 1.2)
        self.typing_speed_variation = response_config.get('typing_speed_variation', 0.3)
        self.typing_min_delay = response_config.get('typing_min_delay', 0.5)
        self.typing_max_delay = response_config.get('typing_max_delay', 2.0)
        
        # 角色信息
        self.character_name = "AI助手"
        self.character_personality = "智能助手"
        
        # 多用戶會話管理
        self.user_sessions: Dict[str, Dict] = {}
        self.max_history_length = 7
        
        # 並發控制
        self.max_concurrent_requests = 5
        self.request_semaphore = asyncio.Semaphore(5)
        
        # 初始化狀態
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化核心服務"""
        if self._initialized:
            return True
            
        try:
            self.logger.info("🚀 初始化 VTuber AI 核心服務...")
            
            # Windows 系統優化
            if sys.platform == "win32":
                optimizer = WindowsOptimizer(self.config)
                optimizer.optimize()
            
            # 初始化 LLM 管理器
            self.llm_manager = LLMManager(self.config)
            await self.llm_manager.initialize()
            
            # 獲取角色信息
            await self._load_character_info()
            
            # 初始化 RAG 系統
            self.rag_system = RAGSystem(self.config, self.llm_manager.embedding_model)
            await self.rag_system.initialize()
            
            # 設置RAG系統引用
            self.llm_manager.set_rag_system_reference(self.rag_system)
            
            # 初始化智慧換行處理器
            self.smart_line_break_filter = SmartLineBreakFilter()
            
            # 初始化 STT 服務（如果啟用）
            if self.stt_enabled:
                await self._initialize_stt_service()
            
            self._initialized = True
            self.logger.info("✅ 核心服務初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"核心服務初始化失敗: {e}")
            return False
    
    async def _load_character_info(self):
        """載入角色信息"""
        try:
            if hasattr(self.llm_manager, 'personality_core') and self.llm_manager.personality_core.core_data:
                identity = self.llm_manager.personality_core.get_character_identity()
                personality = self.llm_manager.personality_core.get_personality_traits()
                
                self.character_name = identity['name'].get('zh', '露西婭')
                self.character_personality = ', '.join(personality['primary_traits'])
                
                self.logger.info(f"✅ 角色信息載入: {self.character_name}")
        except Exception as e:
            self.logger.warning(f"載入角色信息失敗，使用默認值: {e}")
    
    async def _initialize_stt_service(self):
        """初始化 STT 服務"""
        try:
            self.logger.info("🎤 初始化 STT 語音識別服務...")
            
            # 創建 STT 服務
            self.stt_service = await create_stt_service(self.config)
            
            # 註冊 STT 轉錄回調 - 無論是否自動回應都要註冊，因為GUI需要預覽功能
            self.stt_service.add_transcription_callback(self._on_stt_transcription)
            
            # 註冊錯誤回調
            self.stt_service.add_error_callback(self._on_stt_error)
            
            # 註冊停止回調
            self.stt_service.add_stop_callback(self._on_stt_stopped)
            
            self.logger.info("✅ STT 服務初始化完成")
            
        except Exception as e:
            self.logger.error(f"STT 服務初始化失敗: {e}")
            self.stt_service = None
            self.stt_enabled = False
    
    async def _on_stt_transcription(self, result: TranscriptionResult):
        """STT 轉錄結果回調"""
        try:
            # 實時預覽功能 - 即使不是最終結果也要更新GUI
            if self.gui_voice_preview_callback and result.text.strip():
                self.gui_voice_preview_callback(result.text, result.is_final)
            
            # 只處理最終結果
            if not result.is_final or not result.text.strip():
                return
            
            self.logger.info(f"🎤 收到語音輸入: {result.text}")
            
            # 通知GUI語音識別完成
            if self.gui_voice_status_callback:
                self.gui_voice_status_callback(False, f"✅ 識別完成: {result.text[:20]}...")
            
            # 如果啟用自動回應，生成AI回應
            if self.auto_response_enabled:
                # 使用系統用戶ID進行語音對話
                stt_user_id = "stt_user"
                
                # 生成回應
                response_data = await self.generate_response(
                    user_id=stt_user_id,
                    user_input=result.text,
                    context={"source": "voice", "timestamp": result.timestamp.isoformat()}
                )
                
                if response_data.get("success"):
                    self.logger.info(f"🤖 語音回應: {response_data.get('response', '')}")
                else:
                    self.logger.error(f"語音回應生成失敗: {response_data.get('error', '未知錯誤')}")
            
        except Exception as e:
            self.logger.error(f"處理STT轉錄結果失敗: {e}")
            if self.gui_voice_status_callback:
                self.gui_voice_status_callback(False, f"❌ 語音處理錯誤: {str(e)}")
    
    def _on_stt_error(self, error_message: str):
        """STT 錯誤回調"""
        self.logger.error(f"STT 錯誤: {error_message}")
        if self.gui_voice_status_callback:
            self.gui_voice_status_callback(False, f"❌ STT 錯誤: {error_message}")
    
    def _on_stt_stopped(self):
        """STT 停止回調"""
        self.logger.info("🔇 STT 監聽已停止")
        # 通知GUI更新狀態
        if self.gui_voice_status_callback:
            self.gui_voice_status_callback(False, "⏹️ 語音監聽已停止")
        # 調用GUI停止回調（如果有）
        if self.gui_voice_stop_callback:
            try:
                self.gui_voice_stop_callback()
            except Exception as e:
                self.logger.error(f"GUI停止回調執行失敗: {e}")
    
    def set_gui_voice_callbacks(self, preview_callback=None, status_callback=None, stop_callback=None):
        """設置GUI語音回調函數"""
        self.gui_voice_preview_callback = preview_callback
        self.gui_voice_status_callback = status_callback
        self.gui_voice_stop_callback = stop_callback
    
    async def generate_response(self, user_id: str, user_input: str, **kwargs) -> Dict[str, Any]:
        """生成AI回應 - 統一接口"""
        async with self.request_semaphore:
            try:
                if not self._initialized:
                    return {"error": "服務未初始化", "success": False}
                
                # 獲取或創建用戶會話
                session = self._get_or_create_user_session(user_id)
                conversation_history = session['conversation_history']
                
                # 生成回應 - 保持原有的異步調用
                response = await self.llm_manager.generate_response(
                    prompt=user_input,
                    context=kwargs.get('context'),
                    conversation_history=conversation_history,
                    rag_enabled=self.rag_enabled
                )
                
                # 應用智慧換行處理
                if self.line_break_enabled and self.smart_line_break_filter:
                    original_response = response
                    response = self.smart_line_break_filter.filter(
                        response=response,
                        user_input=user_input,
                        context=kwargs.get('context', {})
                    )
                    
                    # 記錄處理統計
                    if response != original_response:
                        self.logger.debug(f"智慧換行處理: 原始 {len(original_response)} 字 -> 處理後 {len(response)} 字")
                
                # 更新會話歷史
                updated_history = self._record_conversation_turn(
                    user_input, response, conversation_history
                )
                session['conversation_history'] = updated_history
                session['last_active'] = datetime.now()
                session['request_count'] = session.get('request_count', 0) + 1
                
                return {
                    "response": response,
                    "character_name": self.character_name,
                    "conversation_length": len(updated_history),
                    "max_length": self.max_history_length,
                    "success": True
                }
                
            except Exception as e:
                self.logger.error(f"生成回應失敗 (用戶 {user_id}): {e}")
                return {"error": str(e), "success": False}
    
    async def generate_response_with_typing(self, user_id: str, user_input: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """生成AI回應 - 支持打字模擬的流式輸出"""
        async with self.request_semaphore:
            try:
                if not self._initialized:
                    yield {"type": "error", "error": "服務未初始化", "success": False}
                    return
                
                # 獲取或創建用戶會話
                session = self._get_or_create_user_session(user_id)
                conversation_history = session['conversation_history']
                
                # 生成回應
                response = await self.llm_manager.generate_response(
                    prompt=user_input,
                    context=kwargs.get('context'),
                    conversation_history=conversation_history,
                    rag_enabled=self.rag_enabled
                )
                
                # 應用智慧換行處理
                if self.line_break_enabled and self.smart_line_break_filter:
                    original_response = response
                    response = self.smart_line_break_filter.filter(
                        response=response,
                        user_input=user_input,
                        context=kwargs.get('context', {})
                    )
                    
                    if response != original_response:
                        self.logger.debug(f"智慧換行處理: 原始 {len(original_response)} 字 -> 處理後 {len(response)} 字")
                
                # 更新會話歷史
                updated_history = self._record_conversation_turn(
                    user_input, response, conversation_history
                )
                session['conversation_history'] = updated_history
                session['last_active'] = datetime.now()
                session['request_count'] = session.get('request_count', 0) + 1
                
                # 如果啟用打字模擬，逐行輸出
                if self.typing_simulation_enabled:
                    # 發送思考開始信號
                    yield {
                        "type": "thinking",
                        "content": "🤔 思考中...",
                        "character_name": self.character_name,
                        "success": True
                    }
                    
                    # 發送回應開始信號
                    yield {
                        "type": "response_start",
                        "character_name": self.character_name,
                        "success": True
                    }
                    
                    # 流式輸出回應內容
                    async for chunk in self._simulate_typing_output(response):
                        chunk.update({
                            "character_name": self.character_name,
                            "conversation_length": len(updated_history),
                            "max_length": self.max_history_length,
                            "success": True
                        })
                        yield chunk
                    
                    # 發送完成信號
                    yield {
                        "type": "response_complete",
                        "character_name": self.character_name,
                        "conversation_length": len(updated_history),
                        "max_length": self.max_history_length,
                        "success": True
                    }
                else:
                    # 不使用打字模擬，直接返回完整回應
                    yield {
                        "type": "response_complete",
                        "response": response,
                        "character_name": self.character_name,
                        "conversation_length": len(updated_history),
                        "max_length": self.max_history_length,
                        "success": True
                    }
                
            except Exception as e:
                self.logger.error(f"生成回應失敗 (用戶 {user_id}): {e}")
                yield {"type": "error", "error": str(e), "success": False}
    
    async def _simulate_typing_output(self, response: str) -> AsyncGenerator[Dict[str, Any], None]:
        """模擬打字輸出效果"""
        lines = response.split('\n')
        
        for i, line in enumerate(lines):
            # 計算這一行的延遲時間
            base_delay = self.typing_speed
            variation = random.uniform(-self.typing_speed_variation, self.typing_speed_variation)
            actual_delay = max(self.typing_min_delay, min(self.typing_max_delay, base_delay + variation))
            
            # 根據行長度微調延遲時間
            line_length = len(line.strip())
            if line_length > 20:
                actual_delay *= 1.2  # 長行多等一點
            elif line_length < 5:
                actual_delay *= 0.8  # 短行少等一點
            
            # 如果不是第一行，先等待
            if i > 0:
                await asyncio.sleep(actual_delay)
            
            # 輸出當前行內容
            if line.strip():  # 只有非空行才輸出
                yield {
                    "type": "response_chunk",
                    "content": line + "\n" if i < len(lines) - 1 else line,
                    "line_number": i + 1,
                    "total_lines": len(lines),
                    "delay_used": actual_delay if i > 0 else 0
                }
            else:
                # 空行也要輸出換行
                yield {
                    "type": "response_chunk",
                    "content": "\n",
                    "line_number": i + 1,
                    "total_lines": len(lines),
                    "delay_used": actual_delay if i > 0 else 0
                }
    
    def _calculate_typing_delay(self, line: str, line_number: int, total_lines: int) -> float:
        """計算單行的打字延遲時間"""
        base_delay = self.typing_speed
        
        # 基於行長度的調整
        line_length = len(line.strip())
        if line_length > 25:
            length_factor = 1.3
        elif line_length > 15:
            length_factor = 1.1
        elif line_length < 5:
            length_factor = 0.7
        else:
            length_factor = 1.0
        
        # 基於行內容的調整
        content_factor = 1.0
        if any(word in line for word in ['？', '！', '?', '!']):
            content_factor = 1.2  # 問句和感嘆句稍慢
        elif any(word in line for word in ['嗯', '啊', '哦', '呢']):
            content_factor = 0.9  # 語氣詞稍快
        
        # 基於位置的調整
        position_factor = 1.0
        if line_number == 1:
            position_factor = 0.8  # 第一行稍快
        elif line_number == total_lines:
            position_factor = 1.1  # 最後一行稍慢
        
        # 添加隨機變化
        variation = random.uniform(-self.typing_speed_variation, self.typing_speed_variation)
        
        # 計算最終延遲
        final_delay = base_delay * length_factor * content_factor * position_factor + variation
        
        # 確保在合理範圍內
        return max(self.typing_min_delay, min(self.typing_max_delay, final_delay))
    
    def _get_or_create_user_session(self, user_id: str) -> Dict:
        """獲取或創建用戶會話"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'conversation_history': [],
                'last_active': datetime.now(),
                'request_count': 0,
                'last_request': datetime.now()
            }
        return self.user_sessions[user_id]
    
    def _record_conversation_turn(self, user_input: str, bot_response: str, 
                                  current_history: List) -> List:
        """記錄對話輪次"""
        new_history = current_history.copy()
        new_history.append((user_input, bot_response))
        
        # 保持歷史長度限制
        if len(new_history) > self.max_history_length:
            new_history = new_history[-self.max_history_length:]
        
        return new_history
    
    # ==================== RAG 功能 ====================
    
    async def add_document(self, file_path: str) -> Dict[str, Any]:
        """添加文檔到知識庫"""
        try:
            if not self._initialized:
                return {"error": "服務未初始化", "success": False}
            
            success = await self.rag_system.add_document(file_path)
            return {
                "success": success,
                "message": f"文檔 {Path(file_path).name} 已成功添加" if success else "文檔添加失敗"
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def search_knowledge_base(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """搜索知識庫"""
        try:
            if not self._initialized:
                return {"error": "服務未初始化", "success": False}
            
            results = await self.rag_system.search(query, top_k=top_k)
            return {
                "results": results,
                "count": len(results),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def clear_knowledge_base(self) -> Dict[str, Any]:
        """清空知識庫"""
        try:
            if not self._initialized:
                return {"error": "服務未初始化", "success": False}
            
            success = await self.rag_system.clear_knowledge_base()
            return {
                "success": success,
                "message": "知識庫已清空" if success else "清空失敗"
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    # ==================== 系統控制 ====================
    
    def toggle_rag(self, enabled: bool) -> Dict[str, Any]:
        """切換RAG狀態"""
        self.rag_enabled = enabled
        return {
            "rag_enabled": self.rag_enabled,
            "message": f"RAG檢索已{'啟用' if enabled else '禁用'}",
            "success": True
        }
    
    def toggle_line_break(self, enabled: bool) -> Dict[str, Any]:
        """切換智慧換行狀態"""
        self.line_break_enabled = enabled
        return {
            "line_break_enabled": self.line_break_enabled,
            "message": f"智慧換行已{'啟用' if enabled else '禁用'}",
            "success": True
        }
    
    # ==================== STT 語音識別控制 ====================
    
    async def toggle_stt(self, enabled: bool) -> Dict[str, Any]:
        """切換STT語音識別狀態"""
        try:
            if enabled and not self.stt_service:
                # 需要初始化STT服務
                await self._initialize_stt_service()
                if not self.stt_service:
                    return {"error": "STT 服務初始化失敗", "success": False}
            
            self.stt_enabled = enabled
            
            if self.stt_service:
                if enabled:
                    success = self.stt_service.start_listening()
                    if not success:
                        return {"error": "STT 服務啟動失敗", "success": False}
                else:
                    success = self.stt_service.stop_listening()
                    if not success:
                        return {"error": "STT 服務停止失敗", "success": False}
            
            return {
                "stt_enabled": self.stt_enabled,
                "message": f"語音識別已{'啟用' if enabled else '禁用'}",
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def toggle_auto_response(self, enabled: bool) -> Dict[str, Any]:
        """切換語音自動回應狀態"""
        self.auto_response_enabled = enabled
        return {
            "auto_response_enabled": self.auto_response_enabled,
            "message": f"語音自動回應已{'啟用' if enabled else '禁用'}",
            "success": True
        }
    
    def get_stt_status(self) -> Dict[str, Any]:
        """獲取STT狀態"""
        try:
            if not self.stt_service:
                return {
                    "stt_available": False,
                    "stt_enabled": False,
                    "is_listening": False,
                    "auto_response_enabled": self.auto_response_enabled,
                    "message": "STT 服務未初始化",
                    "success": True
                }
            
            stats = self.stt_service.get_stats()
            config_info = self.stt_service.get_config_info()
            
            return {
                "stt_available": True,
                "stt_enabled": self.stt_enabled,
                "is_listening": stats.get('is_listening', False),
                "auto_response_enabled": self.auto_response_enabled,
                "config_info": config_info,
                "stats": stats,
                "message": "STT 狀態正常",
                "success": True
            }
        except Exception as e:
            return {
                "stt_available": False,
                "stt_enabled": False,
                "is_listening": False,
                "auto_response_enabled": self.auto_response_enabled,
                "error": str(e),
                "message": f"STT 狀態檢查失敗: {str(e)}",
                "success": False
            }
    
    async def start_stt_listening(self) -> Dict[str, Any]:
        """開始STT監聽（語音按鈕專用）"""
        try:
            if not self.stt_service:
                return {"error": "STT 服務未初始化", "success": False}
            
            if not self.stt_enabled:
                return {"error": "STT 服務未啟用", "success": False}
            
            success = self.stt_service.start_listening()
            if success:
                return {
                    "message": "語音監聽已開始",
                    "success": True
                }
            else:
                return {"error": "STT 監聽啟動失敗", "success": False}
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def stop_stt_listening(self) -> Dict[str, Any]:
        """停止STT監聽（語音按鈕專用）"""
        try:
            if not self.stt_service:
                return {"message": "STT 服務未運行", "success": True}
            
            self.logger.info("🔇 核心服務正在停止STT監聽...")
            
            # 調用STT服務的停止方法
            success = self.stt_service.stop_listening()
            
            if success:
                self.logger.info("✅ STT監聽已成功停止")
                return {
                    "message": "語音監聽已停止",
                    "is_listening": False,
                    "success": True
                }
            else:
                self.logger.error("❌ STT監聽停止失敗")
                return {"error": "STT 監聽停止失敗", "success": False}
                
        except Exception as e:
            self.logger.error(f"停止STT監聽異常: {e}")
            return {"error": str(e), "success": False}
    
    def update_stt_sensitivity(self, silero_sensitivity: float = None, webrtc_sensitivity: int = None) -> Dict[str, Any]:
        """更新STT語音檢測靈敏度"""
        try:
            if not self.stt_service:
                return {"error": "STT 服務未初始化", "success": False}
            
            success = self.stt_service.update_sensitivity(
                silero_sensitivity=silero_sensitivity,
                webrtc_sensitivity=webrtc_sensitivity
            )
            
            return {
                "success": success,
                "message": "STT 靈敏度已更新" if success else "STT 靈敏度更新失敗"
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def toggle_typing_simulation(self, enabled: bool) -> Dict[str, Any]:
        """切換打字模擬狀態"""
        self.typing_simulation_enabled = enabled
        return {
            "typing_simulation_enabled": self.typing_simulation_enabled,
            "message": f"打字模擬已{'啟用' if enabled else '禁用'}",
            "success": True
        }
    
    def set_typing_speed(self, speed: float, variation: float = None) -> Dict[str, Any]:
        """設置打字速度"""
        try:
            if speed < 0.1 or speed > 5.0:
                return {"error": "打字速度必須在 0.1-5.0 秒之間", "success": False}
            
            self.typing_speed = speed
            if variation is not None:
                if variation < 0 or variation > 2.0:
                    return {"error": "速度變化必須在 0-2.0 秒之間", "success": False}
                self.typing_speed_variation = variation
            
            return {
                "typing_speed": self.typing_speed,
                "typing_speed_variation": self.typing_speed_variation,
                "message": f"打字速度已設置為 {speed}±{self.typing_speed_variation} 秒",
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_typing_config(self) -> Dict[str, Any]:
        """獲取打字模擬配置"""
        return {
            "typing_simulation_enabled": self.typing_simulation_enabled,
            "typing_speed": self.typing_speed,
            "typing_speed_variation": self.typing_speed_variation,
            "typing_min_delay": self.typing_min_delay,
            "typing_max_delay": self.typing_max_delay,
            "success": True
        }
    
    def set_typing_preset(self, preset: str) -> Dict[str, Any]:
        """設置打字速度預設"""
        presets = {
            "slow": {"speed": 2.0, "variation": 0.5, "description": "慢速打字（深思熟慮）"},
            "normal": {"speed": 1.2, "variation": 0.3, "description": "正常打字速度"},
            "fast": {"speed": 0.8, "variation": 0.2, "description": "快速打字（活潑）"},
            "very_fast": {"speed": 0.5, "variation": 0.1, "description": "極快打字（興奮）"},
            "thoughtful": {"speed": 1.8, "variation": 0.8, "description": "思考型打字（時快時慢）"}
        }
        
        if preset not in presets:
            return {
                "error": f"未知預設: {preset}",
                "available_presets": list(presets.keys()),
                "success": False
            }
        
        config = presets[preset]
        self.typing_speed = config["speed"]
        self.typing_speed_variation = config["variation"]
        
        return {
            "preset": preset,
            "description": config["description"],
            "typing_speed": self.typing_speed,
            "typing_speed_variation": self.typing_speed_variation,
            "message": f"已套用預設: {config['description']}",
            "success": True
        }
    
    def get_line_break_stats(self) -> Dict[str, Any]:
        """獲取智慧換行統計"""
        try:
            if not self._initialized or not self.smart_line_break_filter:
                return {"error": "智慧換行處理器未初始化", "success": False}
            
            stats = self.smart_line_break_filter.get_stats()
            return {
                "stats": stats,
                "enabled": self.line_break_enabled,
                "filter_name": self.smart_line_break_filter.get_filter_name(),
                "filter_description": self.smart_line_break_filter.get_filter_description(),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取系統統計"""
        try:
            if not self._initialized:
                return {"error": "服務未初始化"}
            
            rag_stats = self.rag_system.get_stats()
            
            # 獲取智慧換行統計
            line_break_stats = {}
            if self.smart_line_break_filter:
                line_break_stats = self.smart_line_break_filter.get_stats()
            
            # 獲取STT統計
            stt_stats = {}
            if self.stt_service:
                stt_stats = self.stt_service.get_stats()
            
            return {
                "total_documents": rag_stats['total_documents'],
                "collection_name": rag_stats['collection_name'],
                "rag_enabled": self.rag_enabled,
                "line_break_enabled": self.line_break_enabled,
                "line_break_stats": line_break_stats,
                "typing_simulation_enabled": self.typing_simulation_enabled,
                "typing_config": {
                    "speed": self.typing_speed,
                    "variation": self.typing_speed_variation,
                    "min_delay": self.typing_min_delay,
                    "max_delay": self.typing_max_delay
                },
                "stt_enabled": self.stt_enabled,
                "stt_available": self.stt_service is not None,
                "stt_stats": stt_stats,
                "auto_response_enabled": self.auto_response_enabled,
                "active_users": len(self.user_sessions),
                "character_name": self.character_name,
                "character_personality": self.character_personality,
                "success": True
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        try:
            if not self._initialized:
                return {"error": "服務未初始化"}
            
            return self.llm_manager.get_model_info()
        except Exception as e:
            return {"error": str(e)}
    
    def clear_user_memory(self, user_id: str) -> Dict[str, Any]:
        """清除用戶記憶"""
        try:
            if user_id in self.user_sessions:
                old_count = len(self.user_sessions[user_id]['conversation_history'])
                self.user_sessions[user_id]['conversation_history'] = []
                return {
                    "success": True,
                    "message": f"已清除 {old_count} 輪對話記憶"
                }
            else:
                return {"success": True, "message": "用戶無對話記憶"}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_user_memory_status(self, user_id: str) -> Dict[str, Any]:
        """獲取用戶記憶狀態"""
        try:
            if user_id in self.user_sessions:
                history = self.user_sessions[user_id]['conversation_history']
                return {
                    "memory_count": len(history),
                    "max_length": self.max_history_length,
                    "last_active": self.user_sessions[user_id]['last_active'].isoformat(),
                    "history": history[-3:] if history else [],  # 返回最近3輪對話
                    "success": True
                }
            else:
                return {
                    "memory_count": 0,
                    "max_length": self.max_history_length,
                    "last_active": None,
                    "history": [],
                    "success": True
                }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    # ==================== 簡繁轉換功能 ====================
    
    def toggle_traditional_chinese(self, enabled: bool) -> Dict[str, Any]:
        """切換簡繁轉換"""
        try:
            if not self._initialized:
                return {"error": "服務未初始化", "success": False}
            
            if hasattr(self.llm_manager, 'response_filter'):
                result = self.llm_manager.response_filter.toggle_traditional_chinese(enabled)
                return {
                    "success": result,
                    "enabled": result if enabled else False,
                    "message": f"簡繁轉換已{'啟用' if result else '禁用' if not enabled else '啟用失敗'}"
                }
            else:
                return {"error": "過濾器未初始化", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_traditional_chinese_status(self) -> Dict[str, Any]:
        """獲取簡繁轉換狀態"""
        try:
            if not self._initialized:
                return {"error": "服務未初始化", "success": False}
            
            if hasattr(self.llm_manager, 'response_filter'):
                status = self.llm_manager.response_filter.get_conversion_status()
                
                # 測試轉換
                test_result = None
                if status['converter_initialized'] and status['conversion_enabled']:
                    test_text = "你好，这是一个测试"
                    converted = self.llm_manager.response_filter.convert_to_traditional_chinese(test_text)
                    test_result = {"original": test_text, "converted": converted}
                
                return {
                    "opencc_available": status['opencc_available'],
                    "converter_initialized": status['converter_initialized'],
                    "conversion_enabled": status['conversion_enabled'],
                    "config_file": status.get('config_file'),
                    "test_result": test_result,
                    "success": True
                }
            else:
                return {"error": "過濾器未初始化", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def cleanup(self):
        """清理資源"""
        try:
            if self.llm_manager:
                self.llm_manager.cleanup()
            
            if self.smart_line_break_filter:
                # 智慧換行處理器通常不需要特殊清理
                pass
            
            if self.stt_service:
                self.stt_service.cleanup()
                self.stt_service = None
            
            self.user_sessions.clear()
            self._initialized = False
            
            self.logger.info("✅ 核心服務資源清理完成")
        except Exception as e:
            self.logger.error(f"核心服務清理失敗: {e}")
    
    # ==================== GUI 相關方法 ====================
    
    async def get_character_info(self) -> Dict[str, Any]:
        """獲取角色信息"""
        try:
            return {
                "success": True,
                "name": self.character_name,
                "personality": getattr(self, 'character_personality', '友善、活潑'),
                "status": "已連接" if self._initialized else "未連接"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"獲取角色信息失敗: {str(e)}"
            }
    
    async def chat(self, message: str, user_id: str) -> Dict[str, Any]:
        """聊天接口（為GUI提供）"""
        try:
            result = await self.generate_response(user_id, message)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"聊天處理失敗: {str(e)}"
            }
    
    async def upload_document(self, file_path: str) -> Dict[str, Any]:
        """上傳文檔到知識庫"""
        try:
            result = await self.add_document(file_path)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"文檔上傳失敗: {str(e)}"
            }
    
    async def search_knowledge(self, query: str) -> Dict[str, Any]:
        """搜索知識庫"""
        try:
            result = await self.search_knowledge_base(query)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"知識搜索失敗: {str(e)}"
            }
    
    async def toggle_traditional_chinese(self, enabled: bool) -> Dict[str, Any]:
        """切換簡繁轉換"""
        try:
            # 這裡可以設置簡繁轉換的邏輯
            return {
                "success": True,
                "enabled": enabled,
                "message": f"簡繁轉換已{'啟用' if enabled else '禁用'}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"簡繁轉換切換失敗: {str(e)}"
            }
    
    async def clear_conversation_memory(self) -> Dict[str, Any]:
        """清除對話記憶"""
        try:
            # 清除所有用戶會話
            cleared_sessions = len(self.user_sessions)
            self.user_sessions.clear()
            
            return {
                "success": True,
                "message": f"已清除 {cleared_sessions} 個會話的記憶"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"清除記憶失敗: {str(e)}"
            }
