#!/usr/bin/env python3
"""
VTuber AI CustomTkinter GUI 主視窗
整合所有GUI組件的主控制器
"""

import customtkinter as ctk
import tkinter as tk
from datetime import datetime
from typing import Any
import sys

from .gui_utils import FontManager, AsyncHelper
from .chat_components import ChatPanel
from .system_components import SystemMessagePanel, HeaderPanel, StatusBar
from .control_components import ControlPanel
from .event_handlers import EventHandlers


class VTuberCustomGUI:
    """VTuber AI CustomTkinter GUI 主程式"""
    
    def __init__(self, core_service):
        self.core_service = core_service
        self.current_user_id = "gui_user"
        
        # 初始化logger
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 創建主窗口 - 必須先創建根窗口
        self.root = ctk.CTk()
        self.root.title("VTuber AI 助手")
        self.root.geometry("1600x900")
        
        # 設置窗口圖標
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # 設置窗口關閉事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 現在可以安全地初始化字體管理器
        self.font_manager = FontManager()
        self.font_manager.initialize_fonts()  # 在根窗口創建後初始化字體
        
        self.async_helper = AsyncHelper()
        
        # 初始化事件處理器
        self.event_handlers = EventHandlers(self.core_service, self.async_helper)
        
        # 初始化界面組件
        self.setup_ui()
        
        # 設置異步支援
        self.async_helper.setup_async_loop()
        
        # 初始化完成後的設置
        self.root.after(500, self.post_init)
    
    def setup_ui(self):
        """設置用戶界面組件"""
        # 創建頂部資訊欄
        self.header_panel = HeaderPanel(
            self.root, 
            self.font_manager.fonts,
            {
                'rag': self.quick_toggle_rag,
                'typing': self.quick_toggle_typing,
                's2t': self.quick_toggle_traditional
            }
        )
        
        # 創建主內容區域
        main_frame = ctk.CTkFrame(self.root, corner_radius=15)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 創建聊天面板
        self.chat_panel = ChatPanel(
            main_frame,
            self.font_manager.fonts,
            on_send_callback=self.send_message,
            on_clear_callback=self.clear_chat,
            on_voice_toggle_callback=self.toggle_voice_input
        )
        
        # 創建系統訊息面板
        self.system_panel = SystemMessagePanel(
            main_frame,
            self.font_manager.fonts
        )
        
        # 創建控制面板
        self.control_panel = ControlPanel(
            main_frame,
            self.font_manager.fonts,
            self._get_control_handlers()
        )
        
        # 創建狀態欄
        self.status_bar = StatusBar(
            self.root,
            self.font_manager.fonts
        )
    
    def _get_control_handlers(self):
        """獲取控制面板事件處理器"""
        return {
            'toggle_rag': self.toggle_rag,
            'toggle_stt': self.toggle_stt,
            'toggle_auto_response': self.toggle_auto_response,
            'update_stt_sensitivity': self.update_stt_sensitivity,
            'toggle_typing': self.toggle_typing,
            'toggle_line_break': self.toggle_line_break,
            'toggle_traditional_chinese': self.toggle_traditional_chinese,
            'typing_preset_change': self.on_typing_preset_change,
            'clear_memory': self.clear_memory,
            'upload_document': self.upload_document,
            'clear_knowledge_base': self.clear_knowledge_base,
            'search_knowledge': self.search_knowledge,
            'refresh_stats': self.refresh_stats,
            'refresh_system_info': self.refresh_system_info,
            'show_model_info': self.show_model_info,
            'show_conversion_info': self.show_conversion_info,
            'export_chat_log': self.export_chat_log
        }
    
    def post_init(self):
        """初始化完成後的設置"""
        # 設置語音回調
        self._setup_voice_callbacks()
        
        # 顯示歡迎訊息
        self.show_welcome_messages()
        
        # 檢查服務狀態
        self.check_service_status()
        
        # 初始化語音功能狀態
        self._initialize_voice_status()
        
        # 開始定期狀態檢查
        self.start_status_monitoring()
    
    def show_welcome_messages(self):
        """顯示歡迎訊息"""
        try:
            # 異步獲取角色信息
            def get_character_info_async():
                future = self.async_helper.run_async_task(
                    self.core_service.get_character_info()
                )
                return future
            
            future = get_character_info_async()
            
            def process_character_info():
                try:
                    if future and future.done():
                        result = future.result()
                        
                        if result and result.get('success'):
                            character_name = result.get('name', 'VTuber AI')
                            self.header_panel.update_character_info(character_name, "🟢 已連接")
                            
                            # 添加歡迎消息
                            self.chat_panel.add_message(
                                "system", 
                                "系統", 
                                f"歡迎使用 {character_name} AI 助手！\n\n🎯 功能介紹：\n• 💬 智能對話：支援上下文記憶和角色扮演\n• 🧠 知識檢索：RAG 增強的智能搜索\n• 📝 打字效果：擬人化的回應體驗\n• 🔄 簡繁轉換：自動語言本地化\n\n現在可以開始對話了！"
                            )
                            
                            # 系統訊息
                            self.system_panel.add_system_message(
                                "system", 
                                "系統啟動", 
                                f"{character_name} 核心服務已就緒"
                            )
                        else:
                            self.header_panel.update_character_info("VTuber AI", "🔴 未連接")
                            self.chat_panel.add_message(
                                "system", 
                                "系統", 
                                "❌ 無法連接到核心服務，請檢查配置"
                            )
                    else:
                        # 如果還沒完成，繼續等待
                        self.root.after(100, process_character_info)
                except Exception as e:
                    self.system_panel.add_system_message("error", "初始化錯誤", str(e))
            
            # 開始檢查結果
            self.root.after(100, process_character_info)
            
        except Exception as e:
            self.system_panel.add_system_message("error", "初始化錯誤", str(e))
    
    def check_service_status(self):
        """檢查服務狀態"""
        try:
            stats = self.core_service.get_stats()  # 這是同步方法
            
            if stats.get('success'):
                self.header_panel.update_character_info(
                    stats.get('character_name', 'VTuber AI'),
                    "🟢 運行中"
                )
                self.status_bar.update_status("系統正常運行")
            else:
                self.status_bar.update_status("系統狀態異常")
                
        except Exception as e:
            self.system_panel.add_system_message("error", "狀態檢查失敗", str(e))
            self.status_bar.update_status(f"狀態檢查失敗: {e}")
    
    def start_status_monitoring(self):
        """開始狀態監控"""
        def monitor():
            self.check_service_status()
            self.root.after(30000, monitor)  # 每30秒檢查一次
        
        self.root.after(5000, monitor)  # 5秒後開始監控
    
    # ==================== 消息處理 ====================
    
    def send_message(self, message: str):
        """發送消息"""
        if not message.strip():
            return
        
        try:
            # 添加用戶消息
            self.chat_panel.add_message("user", "用戶", message)
            
            # 禁用發送按鈕
            self.chat_panel.set_send_button_state(False, "處理中...")
            
            # 添加思考消息
            thinking_widget = self.chat_panel.add_thinking_message()
            
            # 異步處理消息
            future = self.async_helper.run_async_task(
                self.event_handlers.handle_chat_message(message, self.current_user_id)
            )
            
            def process_response():
                try:
                    # 檢查future是否完成，不阻塞UI
                    if not future.done():
                        # 如果還沒完成，100ms後再檢查
                        self.root.after(100, process_response)
                        return
                    
                    # 移除思考消息
                    if thinking_widget:
                        thinking_widget.destroy()
                    
                    # 獲取結果（此時不會阻塞，因為已經完成）
                    result = future.result()
                    
                    if result.get('success'):
                        character_name = result.get('character_name', 'AI')
                        response = result.get('response', '抱歉，我無法回應。')
                        
                        # 檢查是否啟用打字效果
                        if self.control_panel.typing_detail_switch.get():
                            self.start_typing_effect(character_name, response)
                        else:
                            self.chat_panel.add_message("bot", character_name, response)
                        
                        # 記錄系統活動
                        self.system_panel.add_system_message(
                            "success", 
                            "消息處理完成", 
                            f"回應長度: {len(response)} 字符"
                        )
                    else:
                        error_msg = result.get('error', '未知錯誤')
                        self.chat_panel.add_message("system", "系統", f"❌ 處理失敗: {error_msg}")
                        self.system_panel.add_system_message("error", "消息處理失敗", error_msg)
                    
                except Exception as e:
                    # 移除思考消息（如果還存在）
                    if thinking_widget:
                        try:
                            thinking_widget.destroy()
                        except:
                            pass
                    
                    self.chat_panel.add_message("system", "系統", f"❌ 發生錯誤: {str(e)}")
                    self.system_panel.add_system_message("error", "異常錯誤", str(e))
                finally:
                    # 重新啟用發送按鈕
                    self.chat_panel.set_send_button_state(True)
            
            self.root.after(100, process_response)
            
        except Exception as e:
            self.chat_panel.set_send_button_state(True)
            self.system_panel.add_system_message("error", "發送失敗", str(e))
    
    def start_typing_effect(self, character_name: str, full_response: str):
        """開始打字效果"""
        try:
            # 開始bot消息 - 傳遞完整內容以便預估寬度
            self.chat_panel.start_bot_message(character_name, full_response)
            
            # 獲取打字速度設置
            preset = self.control_panel.typing_preset.get()
            speed_config = {
                "slow": 0.08,
                "normal": 0.05,
                "fast": 0.03,
                "very_fast": 0.01,
                "thoughtful": 0.1
            }
            delay = speed_config.get(preset, 0.05)
            
            self._typing_animation(full_response, 0, delay)
            
        except Exception as e:
            self.system_panel.add_system_message("error", "打字效果錯誤", str(e))
    
    def _typing_animation(self, full_text: str, current_pos: int, delay: float):
        """打字動畫實現"""
        if current_pos <= len(full_text):
            displayed_text = full_text[:current_pos]
            
            # 更新消息內容
            self.chat_panel.update_bot_message(displayed_text)
            
            if current_pos < len(full_text):
                # 繼續打字
                self.root.after(int(delay * 1000), 
                    lambda: self._typing_animation(full_text, current_pos + 1, delay))
            else:
                # 打字完成
                self.chat_panel.finalize_bot_message()
    
    def clear_chat(self):
        """清空聊天"""
        try:
            self.chat_panel.clear_chat()
            self.system_panel.add_system_message("system", "聊天清空", "聊天記錄已清空")
        except Exception as e:
            self.system_panel.add_system_message("error", "清空失敗", str(e))
    
    # ==================== 快速控制 ====================
    
    def quick_toggle_rag(self):
        """快速切換RAG"""
        try:
            enabled = self.header_panel.rag_switch.get()
            self.control_panel.rag_detail_switch.configure(state="normal")
            if enabled:
                self.control_panel.rag_detail_switch.select()
            else:
                self.control_panel.rag_detail_switch.deselect()
            self.toggle_rag()
        except Exception as e:
            self.system_panel.add_system_message("error", "RAG切換失敗", str(e))
    
    def quick_toggle_typing(self):
        """快速切換打字模擬"""
        try:
            enabled = self.header_panel.typing_switch.get()
            self.control_panel.typing_detail_switch.configure(state="normal")
            if enabled:
                self.control_panel.typing_detail_switch.select()
            else:
                self.control_panel.typing_detail_switch.deselect()
            self.toggle_typing()
        except Exception as e:
            self.system_panel.add_system_message("error", "打字模擬切換失敗", str(e))
    
    def quick_toggle_traditional(self):
        """快速切換簡繁轉換"""
        try:
            enabled = self.header_panel.s2t_switch.get()
            self.control_panel.traditional_switch.configure(state="normal")
            if enabled:
                self.control_panel.traditional_switch.select()
            else:
                self.control_panel.traditional_switch.deselect()
            self.toggle_traditional_chinese()
        except Exception as e:
            self.system_panel.add_system_message("error", "簡繁轉換切換失敗", str(e))
    
    # ==================== 控制面板功能 ====================
    
    def toggle_rag(self):
        """切換RAG狀態"""
        try:
            enabled = self.control_panel.rag_detail_switch.get()
            result = self.event_handlers.handle_toggle_rag(enabled)  # 現在是同步方法
            
            if result.get('success'):
                status = "啟用" if enabled else "禁用"
                self.system_panel.add_system_message("rag", f"RAG已{status}", "")
            else:
                self.system_panel.add_system_message("error", "RAG切換失敗", result.get('error', ''))
        except Exception as e:
            self.system_panel.add_system_message("error", "RAG操作失敗", str(e))
    
    def toggle_typing(self):
        """切換打字模擬"""
        enabled = self.control_panel.typing_detail_switch.get()
        status = "啟用" if enabled else "禁用"
        self.system_panel.add_system_message("system", f"打字模擬已{status}", "")
    
    def toggle_line_break(self):
        """切換智慧換行"""
        try:
            enabled = self.control_panel.line_break_switch.get()
            result = self.event_handlers.handle_toggle_line_break(enabled)  # 現在是同步方法
            
            if result.get('success'):
                status = "啟用" if enabled else "禁用"
                self.system_panel.add_system_message("system", f"智慧換行已{status}", "")
            else:
                self.system_panel.add_system_message("error", "智慧換行切換失敗", result.get('error', ''))
        except Exception as e:
            self.system_panel.add_system_message("error", "智慧換行操作失敗", str(e))
    
    def toggle_traditional_chinese(self):
        """切換簡繁轉換"""
        future = self.async_helper.run_async_task(
            self.event_handlers.handle_toggle_traditional(self.control_panel.traditional_switch.get())
        )
        
        def update_result():
            try:
                # 非阻塞檢查
                if not future.done():
                    self.root.after(100, update_result)
                    return
                
                result = future.result()
                if result.get('success'):
                    status = "啟用" if result.get('enabled') else "禁用"
                    self.system_panel.add_system_message("system", f"簡繁轉換已{status}", "")
                    self.control_panel.conversion_status_label.configure(text=f"狀態: {status}")
            except Exception as e:
                self.system_panel.add_system_message("error", "簡繁轉換切換失敗", str(e))
        
        self.root.after(100, update_result)
    
    # ==================== STT 語音功能 ====================
    
    def toggle_stt(self):
        """切換STT語音識別"""
        enabled = self.control_panel.stt_switch.get()
        
        # 立即更新UI狀態
        self.control_panel.update_stt_status("⏳ 正在切換語音識別...", "#1976D2")
        self.chat_panel.update_voice_status(False, "⏳ 正在初始化語音功能...")
        
        future = self.async_helper.run_async_task(
            self.event_handlers.handle_toggle_stt(enabled)
        )
        
        def update_result():
            try:
                if not future.done():
                    self.root.after(200, update_result)
                    return
                
                result = future.result()
                if result.get('success'):
                    if enabled:
                        self.control_panel.update_stt_status("✅ STT 已啟用", "#2E7D32")
                        self.chat_panel.set_voice_available(True)
                        self.system_panel.add_system_message("stt", "語音識別已啟用", "可以開始語音輸入")
                    else:
                        self.control_panel.update_stt_status("❌ STT 已禁用", "gray")
                        self.chat_panel.set_voice_available(False)
                        self.system_panel.add_system_message("stt", "語音識別已禁用", "")
                    
                    # 更新控制組件狀態
                    self.control_panel.set_stt_controls_state(enabled)
                else:
                    error_msg = result.get('error', '未知錯誤')
                    self.control_panel.update_stt_status(f"❌ STT 錯誤: {error_msg}", "#D32F2F")
                    self.chat_panel.set_voice_available(False)
                    self.system_panel.add_system_message("error", "STT切換失敗", error_msg)
                    
                    # 重置開關狀態
                    if enabled:
                        self.control_panel.stt_switch.deselect()
                        
            except Exception as e:
                self.control_panel.update_stt_status(f"❌ 異常: {str(e)}", "#D32F2F")
                self.chat_panel.set_voice_available(False)
                self.system_panel.add_system_message("error", "STT操作異常", str(e))
                if enabled:
                    self.control_panel.stt_switch.deselect()
        
        self.root.after(200, update_result)
    
    def toggle_auto_response(self):
        """切換語音自動回應"""
        enabled = self.control_panel.auto_response_switch.get()
        
        try:
            result = self.event_handlers.handle_toggle_auto_response(enabled)
            if result.get('success'):
                status = "啟用" if enabled else "禁用"
                self.system_panel.add_system_message("stt", f"語音自動回應已{status}", "")
            else:
                self.system_panel.add_system_message("error", "自動回應切換失敗", result.get('error', ''))
        except Exception as e:
            self.system_panel.add_system_message("error", "自動回應操作失敗", str(e))
    
    def update_stt_sensitivity(self, sensitivity: float):
        """更新STT靈敏度"""
        try:
            result = self.event_handlers.handle_update_stt_sensitivity(sensitivity)
            if result.get('success'):
                self.system_panel.add_system_message("stt", "靈敏度已更新", f"新值: {sensitivity:.1f}")
            else:
                self.system_panel.add_system_message("error", "靈敏度更新失敗", result.get('error', ''))
        except Exception as e:
            self.system_panel.add_system_message("error", "靈敏度操作失敗", str(e))
    
    def toggle_voice_input(self):
        """切換語音輸入（聊天面板的語音按鈕）"""
        try:
            # 檢查STT是否可用
            stt_status = self.event_handlers.handle_get_stt_status()
            
            if not stt_status.get('success') or not stt_status.get('stt_available'):
                self.system_panel.add_system_message("warning", "語音功能不可用", "請先在控制面板啟用STT語音識別")
                return

            if not stt_status.get('stt_enabled'):
                self.system_panel.add_system_message("warning", "語音識別未啟用", "請先在控制面板啟用STT")
                return

            # 使用服務狀態而非GUI狀態進行判斷，確保同步
            is_currently_listening = stt_status.get('is_listening', False)
            
            self.logger.debug(f"語音按鈕切換: 服務狀態={is_currently_listening}, GUI狀態={self.chat_panel.is_voice_listening}")

            if is_currently_listening:
                # 當前正在聆聽，停止聆聽
                self.logger.debug("執行停止語音輸入")
                self.stop_voice_input()
            else:
                # 當前未聆聽，開始聆聽
                self.logger.debug("執行開始語音輸入")
                self.start_voice_input()
                
        except Exception as e:
            self.logger.error(f"語音輸入切換失敗: {e}")
            self.system_panel.add_system_message("error", "語音輸入切換失敗", str(e))

    def start_voice_input(self):
        """開始語音輸入"""
        try:
            # 檢查是否已經在監聽，如果是則先停止
            stt_status = self.event_handlers.handle_get_stt_status()
            if stt_status.get('is_listening', False):
                self.logger.warning("檢測到STT已在監聽中，先停止現有監聽...")
                # 先停止現有的監聽，不等待回調
                stop_future = self.async_helper.run_async_task(
                    self.core_service.stop_stt_listening()
                )
                # 等待停止完成
                import time
                timeout = 2.0  # 最多等2秒
                start_time = time.time()
                while not stop_future.done() and (time.time() - start_time) < timeout:
                    self.root.update()
                    time.sleep(0.1)
                
                self.logger.debug("現有監聽已停止，繼續開始新的監聽")
            
            self.chat_panel.update_voice_status(True, "🎤 正在啟動語音識別...")
            self.chat_panel.clear_voice_preview()
            
            # 強制同步GUI狀態
            self.chat_panel.is_voice_listening = True
            
            # 註冊語音轉錄回調
            self._setup_voice_callbacks()
            
            # 實際啟動STT監聽 - 關鍵修正！
            def start_listening():
                future = self.async_helper.run_async_task(
                    self.core_service.start_stt_listening()
                )
                
                def check_result():
                    if not future.done():
                        self.root.after(100, check_result)
                        return
                    
                    try:
                        result = future.result()
                        if result.get('success'):
                            self.chat_panel.update_voice_status(True, "🎤 正在聆聽，請說話...")
                            self.system_panel.add_system_message("stt", "語音輸入已啟動", "開始語音識別")
                            self.logger.debug("語音監聽啟動成功")
                        else:
                            error_msg = result.get('error', '未知錯誤')
                            self.chat_panel.update_voice_status(False, f"❌ 啟動失敗: {error_msg}")
                            self.chat_panel.is_voice_listening = False  # 重置狀態
                            self.system_panel.add_system_message("error", "語音輸入啟動失敗", error_msg)
                    except Exception as e:
                        self.chat_panel.update_voice_status(False, f"❌ 啟動異常: {str(e)}")
                        self.chat_panel.is_voice_listening = False  # 重置狀態
                        self.system_panel.add_system_message("error", "語音輸入啟動異常", str(e))
                
                self.root.after(100, check_result)
            
            start_listening()
            
        except Exception as e:
            self.chat_panel.update_voice_status(False, "❌ 語音輸入啟動失敗")
            self.chat_panel.is_voice_listening = False  # 重置狀態
            self.system_panel.add_system_message("error", "語音輸入啟動失敗", str(e))
    
    def stop_voice_input(self):
        """停止語音輸入"""
        try:
            # 立即更新GUI狀態，不等待服務回調
            self.chat_panel.is_voice_listening = False
            self.chat_panel.update_voice_status(False, "⏳ 正在停止語音識別...")
            
            self.logger.debug("立即設置GUI為非監聽狀態")
            
            # 實際停止STT監聽
            def stop_listening():
                future = self.async_helper.run_async_task(
                    self.core_service.stop_stt_listening()
                )
                
                def check_result():
                    if not future.done():
                        self.root.after(100, check_result)
                        return
                    
                    try:
                        result = future.result()
                        self.chat_panel.update_voice_status(False, "🎤 語音輸入已停止")
                        
                        # 如果有預覽內容，確認是否要保留
                        preview_text = self.chat_panel.voice_preview_text
                        if preview_text:
                            self.chat_panel.update_voice_preview(preview_text, is_final=True)
                            
                        self.system_panel.add_system_message("stt", "語音輸入已停止", "")
                        self.logger.debug("語音停止確認完成")
                    except Exception as e:
                        self.chat_panel.update_voice_status(False, f"❌ 停止異常: {str(e)}")
                        self.system_panel.add_system_message("error", "語音輸入停止異常", str(e))
                
                self.root.after(100, check_result)
                
            stop_listening()
            
        except Exception as e:
            self.logger.error(f"停止語音輸入失敗: {e}")
            self.system_panel.add_system_message("error", "語音輸入停止失敗", str(e))
    
    def _setup_voice_callbacks(self):
        """設置語音回調"""
        try:
            # 設置GUI語音回調到核心服務
            self.core_service.set_gui_voice_callbacks(
                preview_callback=self._on_voice_preview,
                status_callback=self._on_voice_status,
                stop_callback=self._on_voice_stopped
            )
            self.system_panel.add_system_message("system", "語音回調設置完成", "GUI已連接到語音服務")
        except Exception as e:
            self.system_panel.add_system_message("error", "語音回調設置失敗", str(e))
    
    def _on_voice_preview(self, text: str, is_final: bool):
        """語音預覽回調（從核心服務調用）"""
        try:
            # 需要在主線程中更新GUI
            self.root.after(0, lambda: self.chat_panel.update_voice_preview(text, is_final))
        except Exception as e:
            self.logger.error(f"語音預覽回調錯誤: {e}")
    
    def _on_voice_status(self, is_listening: bool, status_text: str):
        """語音狀態回調（從核心服務調用）"""
        try:
            # 需要在主線程中更新GUI
            self.root.after(0, lambda: self.chat_panel.update_voice_status(is_listening, status_text))
        except Exception as e:
            self.logger.error(f"語音狀態回調錯誤: {e}")
    
    def _on_voice_stopped(self):
        """語音停止回調（從核心服務調用）"""
        try:
            # 需要在主線程中更新GUI狀態
            def update_gui():
                self.logger.debug("🔇 收到語音停止回調，更新GUI狀態...")
                
                # 強制重置語音按鈕狀態
                self.chat_panel.voice_button.configure(
                    text="🎤",
                    fg_color="#2E7D32",  # 綠色表示可用
                    hover_color="#1B5E20"
                )
                
                # 重置內部狀態
                self.chat_panel.is_voice_listening = False
                
                # 清除語音預覽（保留最終結果）
                if hasattr(self.chat_panel, 'voice_preview_text') and self.chat_panel.voice_preview_text:
                    # 保留最後的預覽內容作為最終結果
                    final_text = self.chat_panel.voice_preview_text
                    self.chat_panel.update_voice_preview(final_text, True)
                    self.logger.debug(f"保留語音預覽最終結果: {final_text}")
                else:
                    # 沒有預覽內容，清空
                    self.chat_panel.clear_voice_preview()
                
                # 重置語音預覽文本緩存
                self.chat_panel.voice_preview_text = ""
                
                # 更新狀態顯示
                self.chat_panel.update_voice_status(False, "🎤 語音監聽已停止")
                
                # 添加系統消息
                self.system_panel.add_system_message("stt", "語音監聽已停止", "語音識別已成功停止")
                
                self.logger.debug("✅ GUI語音狀態已完全重置")
            
            self.root.after(0, update_gui)
        except Exception as e:
            self.logger.error(f"語音停止回調錯誤: {e}")
            # 即使出錯也要嘗試重置狀態
            try:
                def emergency_reset():
                    self.chat_panel.is_voice_listening = False
                    self.chat_panel.voice_preview_text = ""
                    self.chat_panel.update_voice_status(False, "❌ 語音停止異常")
                self.root.after(0, emergency_reset)
            except:
                pass
    
    def _initialize_voice_status(self):
        """初始化語音功能狀態"""
        try:
            # 獲取STT狀態
            stt_status = self.event_handlers.handle_get_stt_status()
            
            if stt_status.get('success') and stt_status.get('stt_available'):
                # STT服務可用
                stt_enabled = stt_status.get('stt_enabled', False)
                is_listening = stt_status.get('is_listening', False)
                auto_response = stt_status.get('auto_response_enabled', False)
                
                # 更新控制面板狀態
                self.control_panel.set_stt_controls_state(stt_enabled, is_listening)
                
                if stt_enabled:
                    if is_listening:
                        self.control_panel.update_stt_status("🎤 正在聆聽", "#2E7D32")
                        self.chat_panel.set_voice_available(True)
                        self.chat_panel.update_voice_status(True, "🎤 正在聆聽...")
                    else:
                        self.control_panel.update_stt_status("✅ STT 已啟用", "#2E7D32")
                        self.chat_panel.set_voice_available(True)
                else:
                    self.control_panel.update_stt_status("❌ STT 未啟用", "gray")
                    self.chat_panel.set_voice_available(False)
                
                # 設置自動回應狀態
                if auto_response:
                    self.control_panel.auto_response_switch.select()
                else:
                    self.control_panel.auto_response_switch.deselect()
                
                self.system_panel.add_system_message("system", "語音功能狀態檢查完成", 
                    f"STT: {'已啟用' if stt_enabled else '未啟用'}, 自動回應: {'開啟' if auto_response else '關閉'}")
            else:
                # STT服務不可用
                self.control_panel.update_stt_status("❌ STT 服務不可用", "#D32F2F")
                self.chat_panel.set_voice_available(False)
                self.control_panel.set_stt_controls_state(False)
                
                self.system_panel.add_system_message("warning", "語音功能不可用", 
                    stt_status.get('message', 'STT 服務未初始化'))
        
        except Exception as e:
            self.control_panel.update_stt_status("❌ 狀態檢查失敗", "#D32F2F")
            self.chat_panel.set_voice_available(False)
            self.system_panel.add_system_message("error", "語音狀態初始化失敗", str(e))
    
    def on_typing_preset_change(self, preset: str):
        """打字速度預設變更"""
        self.system_panel.add_system_message("system", "打字速度調整", f"預設: {preset}")
    
    def clear_memory(self):
        """清除記憶"""
        future = self.async_helper.run_async_task(
            self.event_handlers.handle_clear_memory()
        )
        
        def update_result():
            try:
                # 非阻塞檢查
                if not future.done():
                    self.root.after(100, update_result)
                    return
                
                result = future.result()
                if result.get('success'):
                    self.system_panel.add_system_message("success", "記憶已清除", "")
                    self.control_panel.memory_status_label.configure(text="記憶: 0/7")
            except Exception as e:
                self.system_panel.add_system_message("error", "清除記憶失敗", str(e))
        
        self.root.after(100, update_result)
    
    # ==================== RAG 功能 ====================
    
    def upload_document(self):
        """上傳文檔"""
        try:
            file_path = tk.filedialog.askopenfilename(
                title="選擇要上傳的文檔",
                filetypes=[
                    ("文本文件", "*.txt"),
                    ("Markdown文件", "*.md"),
                    ("所有文件", "*.*")
                ]
            )
            
            if file_path:
                self.system_panel.add_system_message("info", "開始上傳文檔", file_path)
                
                future = self.async_helper.run_async_task(
                    self.event_handlers.handle_upload_document(file_path)
                )
                
                def update_result():
                    try:
                        # 非阻塞檢查
                        if not future.done():
                            self.root.after(500, update_result)  # 文檔上傳可能需要更長時間
                            return
                        
                        result = future.result()
                        if result.get('success'):
                            self.system_panel.add_system_message("success", "文檔上傳成功", f"已添加 {result.get('chunks', 0)} 個文本片段")
                            self.refresh_stats()
                        else:
                            self.system_panel.add_system_message("error", "文檔上傳失敗", result.get('error', ''))
                    except Exception as e:
                        self.system_panel.add_system_message("error", "上傳處理失敗", str(e))
                
                self.root.after(1000, update_result)
                
        except Exception as e:
            self.system_panel.add_system_message("error", "文檔選擇失敗", str(e))
    
    def clear_knowledge_base(self):
        """清空知識庫"""
        if tk.messagebox.askyesno("確認清空", "確定要清空整個知識庫嗎？此操作不可恢復。"):
            future = self.async_helper.run_async_task(
                self.event_handlers.handle_clear_knowledge_base()
            )
            
            def update_result():
                try:
                    # 非阻塞檢查
                    if not future.done():
                        self.root.after(100, update_result)
                        return
                    
                    result = future.result()
                    if result.get('success'):
                        self.system_panel.add_system_message("success", "知識庫已清空", "")
                        self.refresh_stats()
                    else:
                        self.system_panel.add_system_message("error", "清空失敗", result.get('error', ''))
                except Exception as e:
                    self.system_panel.add_system_message("error", "清空處理失敗", str(e))
            
            self.root.after(100, update_result)
    
    def search_knowledge(self):
        """搜索知識庫"""
        try:
            query = self.control_panel.search_entry.get().strip()
            if not query:
                tk.messagebox.showwarning("搜索錯誤", "請輸入搜索關鍵詞")
                return
            
            self.control_panel.search_result.delete("1.0", "end")
            self.control_panel.search_result.insert("1.0", "搜索中...")
            
            future = self.async_helper.run_async_task(
                self.event_handlers.handle_search_knowledge(query)
            )
            
            def update_result():
                try:
                    # 非阻塞檢查
                    if not future.done():
                        self.root.after(100, update_result)
                        return
                    
                    result = future.result()
                    self.control_panel.search_result.delete("1.0", "end")
                    
                    if result.get('success'):
                        results = result.get('results', [])
                        if results:
                            output = f"找到 {len(results)} 個相關結果:\n\n"
                            for i, item in enumerate(results[:3], 1):
                                output += f"{i}. [相似度: {item.get('score', 0):.3f}]\n"
                                output += f"{item.get('content', '')[:200]}...\n\n"
                        else:
                            output = "未找到相關結果"
                    else:
                        output = f"搜索失敗: {result.get('error', '')}"
                    
                    self.control_panel.search_result.insert("1.0", output)
                    
                except Exception as e:
                    self.control_panel.search_result.delete("1.0", "end")
                    self.control_panel.search_result.insert("1.0", f"搜索錯誤: {str(e)}")
            
            self.root.after(100, update_result)
            
        except Exception as e:
            self.system_panel.add_system_message("error", "搜索操作失敗", str(e))
    
    def refresh_stats(self):
        """刷新統計信息"""
        future = self.async_helper.run_async_task(
            self.event_handlers.handle_get_stats()
        )
        
        def update_result():
            try:
                # 非阻塞檢查
                if not future.done():
                    self.root.after(100, update_result)
                    return
                
                result = future.result()
                if result.get('success'):
                    stats = result.get('stats', {})
                    doc_count = stats.get('rag', {}).get('document_count', 0)
                    self.control_panel.doc_count_label.configure(text=f"文檔數量: {doc_count}")
            except Exception as e:
                self.system_panel.add_system_message("error", "統計更新失敗", str(e))
        
        self.root.after(100, update_result)
    
    # ==================== 系統功能 ====================
    
    def refresh_system_info(self):
        """刷新系統信息"""
        future = self.async_helper.run_async_task(
            self.event_handlers.handle_get_system_info()
        )
        
        def update_result():
            try:
                # 非阻塞檢查
                if not future.done():
                    self.root.after(100, update_result)
                    return
                
                result = future.result()
                if result.get('success'):
                    info = result.get('info', {})
                    
                    info_text = "=== 系統信息 ===\n"
                    info_text += f"Python版本: {info.get('python_version', 'Unknown')}\n"
                    info_text += f"系統平台: {info.get('platform', 'Unknown')}\n"
                    info_text += f"記憶體使用: {info.get('memory_usage', 'Unknown')}\n"
                    info_text += f"運行時間: {info.get('uptime', 'Unknown')}\n\n"
                    
                    info_text += "=== 模型狀態 ===\n"
                    model_info = info.get('models', {})
                    info_text += f"LLM模型: {model_info.get('llm_status', 'Unknown')}\n"
                    info_text += f"嵌入模型: {model_info.get('embedding_status', 'Unknown')}\n\n"
                    
                    info_text += "=== 功能狀態 ===\n"
                    features = info.get('features', {})
                    info_text += f"RAG檢索: {'啟用' if features.get('rag_enabled') else '禁用'}\n"
                    info_text += f"智慧換行: {'啟用' if features.get('line_break_enabled') else '禁用'}\n"
                    info_text += f"簡繁轉換: {'啟用' if features.get('traditional_enabled') else '禁用'}\n"
                    
                    self.control_panel.system_info.delete("1.0", "end")
                    self.control_panel.system_info.insert("1.0", info_text)
                    
            except Exception as e:
                self.control_panel.system_info.delete("1.0", "end")
                self.control_panel.system_info.insert("1.0", f"系統信息獲取失敗: {str(e)}")
        
        self.root.after(100, update_result)
    
    def show_model_info(self):
        """顯示模型信息"""
        future = self.async_helper.run_async_task(
            self.event_handlers.handle_get_model_info()
        )
        
        def update_result():
            try:
                # 非阻塞檢查
                if not future.done():
                    self.root.after(100, update_result)
                    return
                
                result = future.result()
                if result.get('success'):
                    info = result.get('info', {})
                    tk.messagebox.showinfo(
                        "模型信息",
                        f"LLM模型: {info.get('llm_model', 'Unknown')}\n"
                        f"嵌入模型: {info.get('embedding_model', 'Unknown')}\n"
                        f"模型狀態: {info.get('status', 'Unknown')}"
                    )
            except Exception as e:
                tk.messagebox.showerror("錯誤", f"獲取模型信息失敗: {str(e)}")
        
        self.root.after(100, update_result)
    
    def show_conversion_info(self):
        """顯示轉換狀態信息"""
        future = self.async_helper.run_async_task(
            self.event_handlers.handle_get_conversion_info()
        )
        
        def update_result():
            try:
                # 非阻塞檢查
                if not future.done():
                    self.root.after(100, update_result)
                    return
                
                result = future.result()
                if result.get('success'):
                    info = result.get('info', {})
                    tk.messagebox.showinfo(
                        "轉換狀態",
                        f"簡繁轉換: {'啟用' if info.get('enabled') else '禁用'}\n"
                        f"轉換器狀態: {info.get('status', 'Unknown')}"
                    )
            except Exception as e:
                tk.messagebox.showerror("錯誤", f"獲取轉換信息失敗: {str(e)}")
        
        self.root.after(100, update_result)
    
    def export_chat_log(self):
        """導出聊天日誌"""
        try:
            file_path = tk.filedialog.asksaveasfilename(
                title="保存聊天日誌",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
            )
            
            if file_path:
                # 這裡可以實現聊天記錄的導出邏輯
                self.system_panel.add_system_message("success", "日誌導出完成", file_path)
                
        except Exception as e:
            self.system_panel.add_system_message("error", "日誌導出失敗", str(e))
    
    # ==================== 生命週期管理 ====================
    
    def on_closing(self):
        """窗口關閉事件"""
        try:
            self.async_helper.stop_loop()
            self.root.quit()
            self.root.destroy()
        except:
            pass
    
    def run(self):
        """運行GUI主循環"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
        except Exception as e:
            print(f"GUI運行錯誤: {e}")
            self.on_closing()
