"""
GUI 工具類和配置
"""
import customtkinter as ctk
import tkinter as tk
import asyncio
import threading
from typing import Dict, Any, Optional

# 設置 CustomTkinter 外觀
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Windows DPI 感知設置
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

class FontManager:
    """字體管理器"""
    
    def __init__(self):
        self.fonts = None  # 延遲初始化
    
    def _setup_fonts(self) -> Dict[str, ctk.CTkFont]:
        """設置字體配置"""
        return {
            'title': ctk.CTkFont(family="Microsoft YaHei UI", size=24, weight="bold"),
            'subtitle': ctk.CTkFont(family="Microsoft YaHei UI", size=16, weight="bold"),
            'body': ctk.CTkFont(family="Microsoft YaHei UI", size=14),
            'body_bold': ctk.CTkFont(family="Microsoft YaHei UI", size=14, weight="bold"),
            'small': ctk.CTkFont(family="Microsoft YaHei UI", size=12),
            'chat_name': ctk.CTkFont(family="Microsoft YaHei UI", size=13, weight="bold"),
            'chat_content': ctk.CTkFont(family="Microsoft YaHei UI", size=14),
            'chat_time': ctk.CTkFont(family="Microsoft YaHei UI", size=10),
            'monospace': ctk.CTkFont(family="Consolas", size=11),
            'input': ctk.CTkFont(family="Microsoft YaHei UI", size=14)
        }
    
    def initialize_fonts(self):
        """在根窗口創建後初始化字體"""
        if self.fonts is None:
            self.fonts = self._setup_fonts()
        return self.fonts

class AsyncHelper:
    """異步操作輔助類"""
    
    def __init__(self):
        self.loop = None
        self.async_thread = None
    
    def setup_async_loop(self):
        """設置異步事件循環"""
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()
    
    def _run_async_loop(self):
        """運行異步事件循環"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def run_async_task(self, coro):
        """運行異步任務"""
        if self.loop:
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future
        return None
    
    def stop_loop(self):
        """停止事件循環"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

class MessageUtils:
    """消息處理工具"""
    
    @staticmethod
    def add_context_menu(textbox, root, fonts):
        """為文字框添加右鍵選單"""
        context_menu = tk.Menu(root, tearoff=0, bg="#2B2B2B", fg="white", 
                              activebackground="#3B82F6", activeforeground="white",
                              font=fonts['small'])
        
        context_menu.add_command(
            label="複製 (Ctrl+C)",
            command=lambda: MessageUtils._copy_selected_text(textbox, root)
        )
        context_menu.add_command(
            label="全選 (Ctrl+A)",
            command=lambda: MessageUtils._select_all_text(textbox)
        )
        context_menu.add_separator()
        context_menu.add_command(
            label="複製整條消息",
            command=lambda: MessageUtils._copy_full_message(textbox, root)
        )
        
        def show_context_menu(event):
            try:
                context_menu.post(event.x_root, event.y_root)
            except:
                pass
        
        textbox.bind("<Button-3>", show_context_menu)
        textbox.bind("<Button-2>", show_context_menu)
        textbox.bind("<Control-c>", lambda e: MessageUtils._copy_selected_text(textbox, root))
        textbox.bind("<Control-a>", lambda e: MessageUtils._select_all_text(textbox))
    
    @staticmethod
    def _copy_selected_text(textbox, root):
        """複製選中的文字"""
        try:
            selected_text = textbox.selection_get()
            if selected_text:
                root.clipboard_clear()
                root.clipboard_append(selected_text)
        except tk.TclError:
            MessageUtils._copy_full_message(textbox, root)
    
    @staticmethod
    def _select_all_text(textbox):
        """全選文字"""
        textbox.configure(state="normal")
        textbox.tag_add("sel", "1.0", "end")
        textbox.configure(state="disabled")
    
    @staticmethod
    def _copy_full_message(textbox, root):
        """複製整條消息"""
        try:
            full_text = textbox.get("1.0", "end-1c")
            root.clipboard_clear()
            root.clipboard_append(full_text)
        except Exception as e:
            print(f"❌ 複製失敗: {e}")
    
    @staticmethod
    def calculate_dynamic_width(content: str, sender_type: str = "bot", min_width: int = 300, max_width: int = 680):
        """根據內容動態計算消息框寬度"""
        try:
            # 優化的字符寬度計算
            char_width = 15  # 進一步增加字符寬度
            padding = 90     # 增加內邊距
            
            # 分析內容 - 考慮中文字符和標點符號
            lines = content.split('\n')
            max_line_length = 0
            
            for line in lines:
                # 中文字符佔用更多空間
                chinese_chars = sum(1 for char in line if '\u4e00' <= char <= '\u9fff')
                english_chars = len(line) - chinese_chars
                # 中文字符按1.3倍計算，英文按1倍
                effective_length = chinese_chars * 1.3 + english_chars
                max_line_length = max(max_line_length, effective_length)
            
            # 計算所需寬度
            content_width = max_line_length * char_width
            total_width = content_width + padding
            
            # 大幅降低最小寬度，特別是用戶消息
            if sender_type == "user":
                min_w, max_w = 150, 450  # 用戶消息大幅降低最小寬度
            elif sender_type == "system":
                min_w, max_w = 200, 500  # 系統消息
            else:  # bot/AI消息
                min_w, max_w = 250, max_width  # AI消息
            
            # 針對超短消息的特殊處理
            content_len = len(content.strip())
            if content_len <= 3:  # 3個字符以內
                min_w = 120  # 非常小的最小寬度
            elif content_len <= 6:  # 6個字符以內
                min_w = min_w * 0.7  # 減少30%
            elif content_len <= 10:  # 10個字符以內
                min_w = min_w * 0.85  # 減少15%
            
            # 限制在合理範圍內
            width = max(min_w, min(total_width, max_w))
            
            return int(width)
            
        except Exception:
            # 降級處理：返回更小的默認寬度
            return {"user": 200, "system": 300, "bot": 350}.get(sender_type, 250)
    
    @staticmethod
    def adjust_textbox_height(textbox, content, actual_width=400):
        """根據實際寬度動態調整文字框高度 - 強制版本"""
        try:
            content_len = len(content.strip())
            
            # 非常激進的高度設置 - 針對短訊息
            if content_len <= 3:
                height = 25  # 極小高度
            elif content_len <= 10:
                height = 30  # 小高度
            elif content_len <= 20:
                height = 35  # 中等高度
            elif content_len <= 50:
                height = 45  # 稍大高度
            else:
                # 長訊息才進行複雜計算
                char_width = 8
                padding = 30
                available_width = actual_width - padding
                chars_per_line = max(15, int(available_width / char_width))
                
                lines = content.split('\n')
                total_lines = 0
                
                for line in lines:
                    if not line.strip():
                        total_lines += 1
                    else:
                        chinese_chars = sum(1 for c in line if '\u4e00' <= c <= '\u9fff')
                        other_chars = len(line) - chinese_chars
                        effective_chars = chinese_chars * 1.5 + other_chars
                        lines_needed = max(1, int((effective_chars + chars_per_line - 1) // chars_per_line))
                        total_lines += lines_needed
                
                height = min(30 + (total_lines * 16), 150)
            
            # 強制設置高度
            textbox.configure(height=int(height))
            
        except Exception as e:
            textbox.configure(height=30)
            
        except Exception:
            # 降級處理
            textbox.configure(height=50)
    
    @staticmethod
    def fix_message_size(message_frame, target_width):
        """修正消息框尺寸"""
        try:
            # 先設置寬度，禁用propagation確保寬度固定
            message_frame.configure(width=target_width)
            message_frame.pack_propagate(False)
            
            # 更新佈局以獲取正確高度
            message_frame.update_idletasks()
            
            # 如果需要，可以調整高度但保持寬度不變
            # 這裡不重新設置寬度，避免覆蓋我們的設置
            
        except Exception as e:
            # 降級處理：至少確保寬度設置
            message_frame.configure(width=target_width)
            message_frame.pack_propagate(False)