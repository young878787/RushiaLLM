"""
聊天相關組件
"""
import customtkinter as ctk
import tkinter as tk
from datetime import datetime
from typing import List, Optional, Callable
from .gui_utils import MessageUtils

class ChatPanel:
    """聊天面板組件"""
    
    def __init__(self, parent, fonts, on_send_callback: Callable, on_clear_callback: Callable):
        self.parent = parent
        self.fonts = fonts
        self.on_send_callback = on_send_callback
        self.on_clear_callback = on_clear_callback
        
        # 消息列表
        self.chat_messages: List = []
        self.current_typing_message = None
        self.thinking_message_widget = None
        self.current_content_textbox = None
        self.current_typing_indicator = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """設置聊天面板UI"""
        # 左側聊天區域
        self.chat_container = ctk.CTkFrame(self.parent)
        self.chat_container.pack(side="left", fill="both", expand=True, padx=(20, 5), pady=20)
        
        # 標題
        chat_header = ctk.CTkLabel(
            self.chat_container,
            text="💬 聊天記錄",
            font=self.fonts['subtitle']
        )
        chat_header.pack(pady=(10, 5))
        
        # 聊天框架
        self.chat_frame = ctk.CTkScrollableFrame(self.chat_container, corner_radius=10)
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # 輸入區域
        self._create_input_area()
    
    def _create_input_area(self):
        """創建輸入區域"""
        input_frame = ctk.CTkFrame(self.chat_container, height=120, corner_radius=10)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        input_frame.pack_propagate(False)
        
        # 輸入框
        self.message_input = ctk.CTkTextbox(
            input_frame,
            height=80,
            font=self.fonts['input'],
            corner_radius=8,
            wrap="word"
        )
        self.message_input.pack(side="left", fill="both", expand=True, padx=(15, 10), pady=15)
        
        # 按鈕區域
        button_frame = ctk.CTkFrame(input_frame, width=100, fg_color="transparent")
        button_frame.pack(side="right", fill="y", padx=(0, 15), pady=15)
        button_frame.pack_propagate(False)
        
        self.send_button = ctk.CTkButton(
            button_frame,
            text="發送",
            command=self._on_send,
            font=self.fonts['body_bold'],
            height=35
        )
        self.send_button.pack(fill="x", pady=(0, 5))
        
        self.clear_chat_button = ctk.CTkButton(
            button_frame,
            text="清空",
            command=self._on_clear,
            font=self.fonts['body'],
            height=35,
            fg_color="gray",
            hover_color="darkgray"
        )
        self.clear_chat_button.pack(fill="x")
        
        # 綁定快捷鍵
        self.message_input.bind("<Return>", self._on_enter_key)
        self.message_input.bind("<Shift-Return>", self._on_shift_enter)
        self.message_input.bind("<Control-Return>", lambda e: self._on_send())
    
    def _on_send(self):
        """發送按鈕處理"""
        message = self.message_input.get("1.0", "end-1c").strip()
        if message and self.on_send_callback:
            self.message_input.delete("1.0", "end")
            self.on_send_callback(message)
    
    def _on_clear(self):
        """清空按鈕處理"""
        if self.on_clear_callback:
            self.on_clear_callback()
    
    def _on_enter_key(self, event):
        """處理Enter鍵"""
        self._on_send()
        return "break"
    
    def _on_shift_enter(self, event):
        """處理Shift+Enter"""
        return None
    
    def add_message(self, sender_type: str, sender_name: str, content: str):
        """添加消息到聊天"""
        colors = {
            "user": "#3B82F6",
            "bot": "#8B5CF6", 
            "system": "#6B7280"
        }
        
        # 創建消息容器 - 不要讓其垂直擴展
        message_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        message_container.pack(fill="x", pady=2, expand=False)  # 改為expand=False，減少pady
        
        # 創建消息框架
        message_frame = ctk.CTkFrame(
            message_container,
            fg_color=colors.get(sender_type, "gray"),
            corner_radius=15
        )
        
        # 動態計算寬度
        width = MessageUtils.calculate_dynamic_width(content, sender_type)
        
        # 設置對齊和寬度 - 移除ipady避免額外高度
        if sender_type == "user":
            message_frame.configure(width=width)
            message_frame.pack_propagate(False)  # 立即禁用propagation
            message_frame.pack(side="right", padx=(30, 15), pady=0)  # 移除ipady
        elif sender_type == "system":
            message_frame.configure(width=width)
            message_frame.pack_propagate(False)  # 立即禁用propagation
            message_frame.pack(anchor="center", padx=80, pady=0)  # 移除ipady
        else:  # bot - AI回覆框
            message_frame.configure(width=width)
            message_frame.pack_propagate(False)  # 立即禁用propagation
            message_frame.pack(side="left", padx=(15, 30), pady=0)  # 移除ipady
        
        # 修正尺寸
        self.parent.after(1, lambda: MessageUtils.fix_message_size(message_frame, width))
        
        # 消息頭部
        self._create_message_header(message_frame, sender_name, sender_type)
        
        # 消息內容
        self._create_message_content(message_frame, content, width)
        
        self.chat_messages.append(message_container)
        self.parent.after(100, self.scroll_to_bottom)
    
    def _create_message_header(self, message_frame, sender_name, sender_type):
        """創建消息頭部"""
        header_frame = ctk.CTkFrame(message_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(10, 5))
        
        sender_label = ctk.CTkLabel(
            header_frame,
            text=sender_name,
            font=self.fonts['chat_name'],
            text_color="white"
        )
        
        time_label = ctk.CTkLabel(
            header_frame,
            text=datetime.now().strftime("%H:%M:%S"),
            font=self.fonts['chat_time'],
            text_color="lightgray"
        )
        
        if sender_type == "user":
            time_label.pack(side="left")
            sender_label.pack(side="right")
        else:
            sender_label.pack(side="left")
            time_label.pack(side="right")
    
    def _create_message_content(self, message_frame, content, width):
        """創建消息內容"""
        # 先計算需要的高度
        from .gui_utils import MessageUtils
        
        content_textbox = ctk.CTkTextbox(
            message_frame,
            font=self.fonts['chat_content'],
            text_color="white",
            fg_color="transparent",
            corner_radius=0,
            border_width=0,
            wrap="word",
            height=35,  # 設置一個合理的初始高度
            activate_scrollbars=False
        )
        content_textbox.pack(fill="x", expand=False, padx=10, pady=8)  # 只填充寬度，不填充高度
        
        content_textbox.insert("1.0", content)
        content_textbox.configure(state="disabled")
        
        # 調整高度和添加右鍵選單
        MessageUtils.adjust_textbox_height(content_textbox, content, width)
        MessageUtils.add_context_menu(content_textbox, self.parent, self.fonts)
    
    def add_thinking_message(self):
        """添加思考消息"""
        message_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        message_container.pack(fill="x", pady=5)
        
        message_frame = ctk.CTkFrame(
            message_container,
            fg_color="#6B7280",
            corner_radius=15
        )
        message_frame.pack(anchor="center", padx=80, pady=0, ipadx=10, ipady=5)
        
        content_label = ctk.CTkLabel(
            message_frame,
            text="🤔 AI正在思考...",
            font=self.fonts['chat_content'],
            text_color="white"
        )
        content_label.pack(fill="x", padx=15, pady=10)
        
        self.parent.after(100, self.scroll_to_bottom)
        return message_container
    
    def start_bot_message(self, character_name: str, full_content: str = ""):
        """開始bot消息（打字效果）"""
        message_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        message_container.pack(fill="x", pady=5, expand=True)
        
        # 動態計算寬度 - 基於完整內容預估
        if full_content:
            width = MessageUtils.calculate_dynamic_width(full_content, "bot")
        else:
            width = 400  # 默認較小寬度，後續會調整
        
        message_frame = ctk.CTkFrame(
            message_container,
            fg_color="#8B5CF6",
            corner_radius=15
        )
        message_frame.configure(width=width)
        message_frame.pack_propagate(False)  # 立即禁用propagation
        message_frame.pack(side="left", padx=(15, 30), pady=0, ipadx=10, ipady=5)
        
        # 消息頭部
        header_frame = ctk.CTkFrame(message_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(10, 5))
        
        sender_label = ctk.CTkLabel(
            header_frame,
            text=character_name,
            font=self.fonts['chat_name'],
            text_color="white"
        )
        sender_label.pack(side="left")
        
        typing_indicator = ctk.CTkLabel(
            header_frame,
            text="⌨️ 正在輸入...",
            font=self.fonts['chat_time'],
            text_color="lightgray"
        )
        typing_indicator.pack(side="right")
        
        # 內容區域
        self.current_content_textbox = ctk.CTkTextbox(
            message_frame,
            font=self.fonts['chat_content'],
            text_color="white",
            fg_color="transparent",
            corner_radius=0,
            border_width=0,
            wrap="word",
            height=35,  # 設置合理的初始高度
            activate_scrollbars=False
        )
        self.current_content_textbox.pack(fill="x", expand=False, padx=10, pady=8)  # 只填充寬度，不填充高度
        self.current_content_textbox.configure(state="normal")
        
        MessageUtils.add_context_menu(self.current_content_textbox, self.parent, self.fonts)
        
        self.chat_messages.append(message_container)
        self.current_typing_message = message_frame
        self.current_typing_indicator = typing_indicator
        self.scroll_to_bottom()
    
    def update_bot_message(self, content: str):
        """更新bot消息內容"""
        if hasattr(self, 'current_content_textbox') and self.current_content_textbox:
            self.current_content_textbox.delete("1.0", "end")
            self.current_content_textbox.insert("1.0", content)
            
            # 動態調整消息框寬度
            if hasattr(self, 'current_typing_message') and self.current_typing_message:
                new_width = MessageUtils.calculate_dynamic_width(content, "bot")
                self.current_typing_message.configure(width=new_width)
            
            # 調整高度
            MessageUtils.adjust_textbox_height(self.current_content_textbox, content, new_width if 'new_width' in locals() else 600)
            self.scroll_to_bottom()
    
    def finalize_bot_message(self):
        """完成bot消息"""
        if hasattr(self, 'current_typing_indicator') and self.current_typing_indicator:
            self.current_typing_indicator.configure(text=datetime.now().strftime("%H:%M:%S"))
        
        if hasattr(self, 'current_content_textbox') and self.current_content_textbox:
            self.current_content_textbox.configure(state="disabled")
        
        self.current_typing_message = None
        self.current_content_textbox = None
    
    def clear_chat(self):
        """清空聊天"""
        for message_container in self.chat_messages:
            message_container.destroy()
        self.chat_messages.clear()
    
    def scroll_to_bottom(self):
        """滾動到底部"""
        try:
            self.chat_frame._parent_canvas.yview_moveto(1.0)
        except:
            pass
    
    def set_send_button_state(self, enabled: bool, text: str = "發送"):
        """設置發送按鈕狀態"""
        state = "normal" if enabled else "disabled"
        self.send_button.configure(state=state, text=text)