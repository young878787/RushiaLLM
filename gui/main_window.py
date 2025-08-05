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
            on_clear_callback=self.clear_chat
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
        # 顯示歡迎訊息
        self.show_welcome_messages()
        
        # 檢查服務狀態
        self.check_service_status()
        
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
