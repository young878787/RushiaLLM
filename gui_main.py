#!/usr/bin/env python3
"""
VTuber AI CustomTkinter GUI 主程式
基於現有核心服務的現代化圖形界面
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import asyncio
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys

# 設置 CustomTkinter 外觀
ctk.set_appearance_mode("dark")  # "light" 或 "dark"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Windows DPI 感知設置（提高字體清晰度）
try:
    import ctypes
    # 設置DPI感知
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

class VTuberCustomGUI:
    def __init__(self, core_service):
        self.core_service = core_service
        self.current_user_id = "gui_user"
        
        # 創建主窗口
        self.root = ctk.CTk()
        self.root.title("VTuber AI 助手")
        self.root.geometry("1600x900")  # 增加寬度以適應三分割布局
        
        # 設置字體 - 使用更清晰的字體
        self.setup_fonts()
        
        # 設置窗口圖標
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # 初始化變數
        self.chat_messages = []
        self.system_messages = []  # 系統訊息列表
        self.current_typing_message = None
        self.thinking_message_widget = None  # 思考消息組件引用
        
        # 初始化界面和異步支援
        self.setup_ui()
        self.setup_async_loop()
        
        # 添加歡迎系統訊息
        self.root.after(500, self.show_welcome_messages)
    
    def setup_fonts(self):
        """設置字體配置"""
        # 定義各種用途的字體
        self.fonts = {
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
    
    def setup_ui(self):
        """設置用戶界面"""
        # 創建頂部信息欄
        self.create_header()
        
        # 創建主內容區域
        self.create_main_content()
        
        # 創建底部狀態欄
        self.create_status_bar()
    
    def create_header(self):
        """創建頂部信息欄"""
        header_frame = ctk.CTkFrame(self.root, height=80, corner_radius=15)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        header_frame.pack_propagate(False)
        
        # 角色頭像和信息
        info_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        
        self.character_name_label = ctk.CTkLabel(
            info_frame,
            text="載入中...",
            font=self.fonts['title']
        )
        self.character_name_label.pack(anchor="w")
        
        self.character_status_label = ctk.CTkLabel(
            info_frame,
            text="🔴 未連接",
            font=self.fonts['body']
        )
        self.character_status_label.pack(anchor="w")
        
        # 右側快速控制
        control_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        control_frame.pack(side="right", padx=20, pady=10)
        
        # RAG 快速開關
        self.rag_switch = ctk.CTkSwitch(
            control_frame,
            text="RAG檢索",
            command=self.quick_toggle_rag,
            font=self.fonts['small']
        )
        self.rag_switch.pack(side="top", pady=2)
        self.rag_switch.select()
        
        # 打字模擬快速開關
        self.typing_switch = ctk.CTkSwitch(
            control_frame,
            text="打字模擬",
            command=self.quick_toggle_typing,
            font=self.fonts['small']
        )
        self.typing_switch.pack(side="top", pady=2)
        self.typing_switch.select()
        
        # 簡繁轉換快速開關
        self.s2t_switch = ctk.CTkSwitch(
            control_frame,
            text="簡繁轉換",
            command=self.quick_toggle_traditional,
            font=self.fonts['small']
        )
        self.s2t_switch.pack(side="top", pady=2)
        self.s2t_switch.select()
    
    def create_main_content(self):
        """創建主內容區域"""
        # 主內容框架
        main_frame = ctk.CTkFrame(self.root, corner_radius=15)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 創建三分割布局：聊天區 + 系統訊息區 + 控制面板
        # 左側聊天區域 (50%)
        chat_container = ctk.CTkFrame(main_frame)
        chat_container.pack(side="left", fill="both", expand=True, padx=(20, 5), pady=20)
        
        # 聊天記錄顯示
        chat_header = ctk.CTkLabel(
            chat_container,
            text="💬 聊天記錄",
            font=self.fonts['subtitle']
        )
        chat_header.pack(pady=(10, 5))
        
        # 聊天框架 - 確保有足夠寬度
        self.chat_frame = ctk.CTkScrollableFrame(chat_container, corner_radius=10)
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # 輸入區域
        self.create_input_area(chat_container)
        
        # 中間系統訊息區域 (25%)
        self.create_system_messages_area(main_frame)
        
        # 右側控制面板 (25%)
        self.create_control_panel(main_frame)
    
    def create_input_area(self, parent):
        """創建輸入區域"""
        input_frame = ctk.CTkFrame(parent, height=120, corner_radius=10)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        input_frame.pack_propagate(False)
        
        # 輸入框 - 移除提示文字，擴大寬度
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
            command=self.send_message,
            font=self.fonts['body_bold'],
            height=35
        )
        self.send_button.pack(fill="x", pady=(0, 5))
        
        self.clear_chat_button = ctk.CTkButton(
            button_frame,
            text="清空",
            command=self.clear_chat,
            font=self.fonts['body'],
            height=35,
            fg_color="gray",
            hover_color="darkgray"
        )
        self.clear_chat_button.pack(fill="x")
        
        # 綁定快捷鍵
        self.message_input.bind("<Return>", self.on_enter_key)
        self.message_input.bind("<Shift-Return>", self.on_shift_enter)
        self.message_input.bind("<Control-Return>", lambda e: self.send_message())
    
    def create_system_messages_area(self, parent):
        """創建系統訊息區域"""
        system_container = ctk.CTkFrame(parent, width=300)
        system_container.pack(side="left", fill="y", padx=(5, 5), pady=20)
        system_container.pack_propagate(False)
        
        # 系統訊息標題
        system_header = ctk.CTkLabel(
            system_container,
            text="📋 系統訊息",
            font=self.fonts['subtitle']
        )
        system_header.pack(pady=(10, 5))
        
        # 系統訊息類型選擇
        type_frame = ctk.CTkFrame(system_container, fg_color="transparent")
        type_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        self.message_type_var = ctk.StringVar(value="all")
        self.message_filter = ctk.CTkOptionMenu(
            type_frame,
            values=["all", "system", "rag", "terminal", "error", "success"],
            variable=self.message_type_var,
            command=self.filter_system_messages,
            font=self.fonts['small']
        )
        self.message_filter.pack(side="left", fill="x", expand=True)
        
        # 清空系統訊息按鈕
        clear_system_btn = ctk.CTkButton(
            type_frame,
            text="清空",
            command=self.clear_system_messages,
            font=self.fonts['small'],
            width=50,
            height=25,
            fg_color="gray",
            hover_color="darkgray"
        )
        clear_system_btn.pack(side="right", padx=(5, 0))
        
        # 系統訊息滾動區域
        self.system_frame = ctk.CTkScrollableFrame(
            system_container, 
            corner_radius=10,
            label_text="最新訊息"
        )
        self.system_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # 系統訊息統計
        self.system_stats_label = ctk.CTkLabel(
            system_container,
            text="訊息數量: 0",
            font=self.fonts['small']
        )
        self.system_stats_label.pack(pady=(0, 10))
    
    def create_control_panel(self, parent):
        """創建控制面板"""
        control_frame = ctk.CTkFrame(parent, width=250, corner_radius=15)
        control_frame.pack(side="right", fill="y", padx=(0, 20), pady=20)
        control_frame.pack_propagate(False)
        
        # 控制面板標題
        title_label = ctk.CTkLabel(
            control_frame,
            text="🎛️ 控制面板",
            font=self.fonts['subtitle']
        )
        title_label.pack(pady=(20, 10))
        
        # 創建分頁
        self.tabview = ctk.CTkTabview(control_frame, width=220, height=580)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)
        
        # AI設置分頁
        self.setup_ai_tab()
        
        # 知識庫分頁
        self.setup_rag_tab()
        
        # 系統分頁
        self.setup_system_tab()
    
    def setup_ai_tab(self):
        """設置AI控制分頁"""
        ai_tab = self.tabview.add("AI設置")
        
        # RAG控制區域
        rag_frame = ctk.CTkFrame(ai_tab, corner_radius=10)
        rag_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            rag_frame,
            text="🧠 智能檢索",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.rag_detail_switch = ctk.CTkSwitch(
            rag_frame,
            text="啟用 RAG 檢索",
            command=self.toggle_rag
        )
        self.rag_detail_switch.pack(pady=10)
        self.rag_detail_switch.select()
        
        # 打字模擬控制
        typing_frame = ctk.CTkFrame(ai_tab, corner_radius=10)
        typing_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            typing_frame,
            text="⌨️ 打字模擬",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.typing_detail_switch = ctk.CTkSwitch(
            typing_frame,
            text="啟用打字效果",
            command=self.toggle_typing
        )
        self.typing_detail_switch.pack(pady=5)
        self.typing_detail_switch.select()
        
        # 打字速度預設
        ctk.CTkLabel(typing_frame, text="速度預設:").pack(pady=(10, 0))
        self.typing_preset = ctk.CTkOptionMenu(
            typing_frame,
            values=["slow", "normal", "fast", "very_fast", "thoughtful"],
            command=self.on_typing_preset_change
        )
        self.typing_preset.pack(pady=(5, 15))
        self.typing_preset.set("normal")
        
        # 智慧換行控制
        line_break_frame = ctk.CTkFrame(ai_tab, corner_radius=10)
        line_break_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            line_break_frame,
            text="📝 智慧換行",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.line_break_switch = ctk.CTkSwitch(
            line_break_frame,
            text="啟用智慧換行",
            command=self.toggle_line_break
        )
        self.line_break_switch.pack(pady=(5, 15))
        self.line_break_switch.select()
        
        # 簡繁轉換控制
        conversion_frame = ctk.CTkFrame(ai_tab, corner_radius=10)
        conversion_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            conversion_frame,
            text="🔄 簡繁轉換",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.traditional_switch = ctk.CTkSwitch(
            conversion_frame,
            text="啟用簡繁轉換",
            command=self.toggle_traditional_chinese
        )
        self.traditional_switch.pack(pady=5)
        self.traditional_switch.select()
        
        # 簡繁轉換狀態顯示
        self.conversion_status_label = ctk.CTkLabel(
            conversion_frame,
            text="狀態: 檢查中...",
            font=self.fonts['small']
        )
        self.conversion_status_label.pack(pady=(5, 15))
        
        # 記憶管理
        memory_frame = ctk.CTkFrame(ai_tab, corner_radius=10)
        memory_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            memory_frame,
            text="🧮 記憶管理",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.memory_status_label = ctk.CTkLabel(
            memory_frame,
            text="記憶: 0/7",
            font=self.fonts['body']
        )
        self.memory_status_label.pack(pady=5)
        
        ctk.CTkButton(
            memory_frame,
            text="清除記憶",
            command=self.clear_memory,
            fg_color="darkred",
            hover_color="red"
        ).pack(pady=(5, 15))
    
    def setup_rag_tab(self):
        """設置知識庫分頁"""
        rag_tab = self.tabview.add("知識庫")
        
        # 文檔管理
        doc_frame = ctk.CTkFrame(rag_tab, corner_radius=10)
        doc_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            doc_frame,
            text="📚 文檔管理",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 10))
        
        ctk.CTkButton(
            doc_frame,
            text="📁 上傳文檔",
            command=self.upload_document
        ).pack(fill="x", padx=15, pady=5)
        
        ctk.CTkButton(
            doc_frame,
            text="🗑️ 清空知識庫",
            command=self.clear_knowledge_base,
            fg_color="darkred",
            hover_color="red"
        ).pack(fill="x", padx=15, pady=(5, 15))
        
        # 搜索測試
        search_frame = ctk.CTkFrame(rag_tab, corner_radius=10)
        search_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            search_frame,
            text="🔍 搜索測試",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 10))
        
        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="輸入搜索詞..."
        )
        self.search_entry.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkButton(
            search_frame,
            text="搜索",
            command=self.search_knowledge
        ).pack(fill="x", padx=15, pady=(5, 10))
        
        # 搜索結果
        self.search_result = ctk.CTkTextbox(
            search_frame,
            height=150,
            font=self.fonts['small']
        )
        self.search_result.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # 統計信息
        stats_frame = ctk.CTkFrame(rag_tab, corner_radius=10)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            stats_frame,
            text="📊 統計信息",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 10))
        
        self.doc_count_label = ctk.CTkLabel(
            stats_frame,
            text="文檔數量: 0",
            font=self.fonts['body']
        )
        self.doc_count_label.pack(pady=5)
        
        ctk.CTkButton(
            stats_frame,
            text="刷新統計",
            command=self.refresh_stats
        ).pack(fill="x", padx=15, pady=(5, 15))
    
    def setup_system_tab(self):
        """設置系統分頁"""
        system_tab = self.tabview.add("系統")
        
        # 系統信息
        info_frame = ctk.CTkFrame(system_tab, corner_radius=10)
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            info_frame,
            text="💻 系統信息",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 10))
        
        self.system_info = ctk.CTkTextbox(
            info_frame,
            height=200,
            font=self.fonts['monospace']
        )
        self.system_info.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        # 控制按鈕
        button_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkButton(
            button_frame,
            text="刷新信息",
            command=self.refresh_system_info,
            width=80
        ).pack(side="left", padx=(0, 5))
        
        ctk.CTkButton(
            button_frame,
            text="模型信息",
            command=self.show_model_info,
            width=80
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="轉換狀態",
            command=self.show_conversion_info,
            width=80
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="導出日誌",
            command=self.export_chat_log,
            width=80
        ).pack(side="right")
    
    def create_status_bar(self):
        """創建狀態欄"""
        status_frame = ctk.CTkFrame(self.root, height=40, corner_radius=10)
        status_frame.pack(fill="x", padx=20, pady=(0, 20))
        status_frame.pack_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="準備中...",
            font=self.fonts['body']
        )
        self.status_label.pack(side="left", padx=20, pady=10)
        
        self.time_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=self.fonts['body']
        )
        self.time_label.pack(side="right", padx=20, pady=10)
        
        # 開始時間更新
        self.update_time()
    
    def setup_async_loop(self):
        """設置異步事件循環"""
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.async_thread.start()
        
        # 定期檢查服務狀態
        self.check_service_status()
    
    def run_async_loop(self):
        """運行異步事件循環"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def run_async_task(self, coro):
        """運行異步任務"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future
    
    # ==================== 消息處理 ====================
    
    def add_message(self, sender_type: str, sender_name: str, content: str):
        """添加消息到聊天"""
        colors = {
            "user": "#3B82F6",
            "bot": "#8B5CF6", 
            "system": "#6B7280"
        }
        
        # 創建消息容器（用於控制對齊）- 添加 expand=True
        message_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        message_container.pack(fill="x", pady=5, expand=True)
        
        # 創建消息框架
        message_frame = ctk.CTkFrame(
            message_container,
            fg_color=colors.get(sender_type, "gray"),
            corner_radius=15
        )
        
        # 根據發送者類型設置對齊方式和寬度
        if sender_type == "user":
            # 用戶消息：右對齊，固定寬度但允許高度自適應
            message_frame.configure(width=450)  # 明確設定寬度
            message_frame.pack(side="right", padx=(50, 20), pady=0, ipadx=10, ipady=5)
            # 使用 winfo_reqwidth() 來保持寬度但允許高度變化
            self.root.after(1, lambda: self._fix_message_size(message_frame, 450))
        elif sender_type == "system":
            # 系統消息：同時顯示在聊天區和系統訊息區
            message_frame.configure(width=500)  # 系統消息稍小一些
            message_frame.pack(anchor="center", padx=100, pady=0, ipadx=15, ipady=5)
            self.root.after(1, lambda: self._fix_message_size(message_frame, 500))
            
            # 同時添加到系統訊息區域
            self.add_system_message("system", sender_name, content)
        else:
            # AI消息：左對齊，固定寬度但允許高度自適應 - 比用戶消息稍寬一些
            message_frame.configure(width=550)  # 明確設定寬度，比用戶消息寬100px
            message_frame.pack(side="left", padx=(20, 50), pady=0, ipadx=25, ipady=5)
            # 使用 winfo_reqwidth() 來保持寬度但允許高度變化
            self.root.after(1, lambda: self._fix_message_size(message_frame, 550))
        
        # 消息頭部
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
            # 用戶消息：時間在左，名字在右
            time_label.pack(side="left")
            sender_label.pack(side="right")
        else:
            # AI/系統消息：名字在左，時間在右
            sender_label.pack(side="left")
            time_label.pack(side="right")
        
        # 消息內容 - 使用 CTkTextbox 支持文字選擇和複製
        content_textbox = ctk.CTkTextbox(
            message_frame,
            font=self.fonts['chat_content'],
            text_color="white",
            fg_color="transparent",  # 透明背景與消息框融合
            corner_radius=0,
            border_width=0,
            wrap="word",
            height=1,  # 初始高度
            activate_scrollbars=False  # 禁用滾動條
        )
        content_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # 插入內容並設置為只讀
        content_textbox.insert("1.0", content)
        content_textbox.configure(state="disabled")  # 只讀，但仍可選擇文字
        
        # 動態調整高度以適應內容
        self._adjust_textbox_height(content_textbox, content)
        
        # 在調整textbox高度後，重新計算消息框高度
        self.root.after(10, lambda: self._recalculate_message_height(message_frame, sender_type))
        
        # 為文字框添加右鍵選單
        self._add_context_menu(content_textbox)
        
        self.chat_messages.append(message_container)
        
        # 自動滾動到底部
        self.root.after(100, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        """滾動到底部"""
        try:
            self.chat_frame._parent_canvas.yview_moveto(1.0)
        except:
            pass
    
    def add_system_message(self, message_type: str, title: str, content: str, level: str = "info"):
        """添加系統訊息"""
        try:
            # 訊息類型圖標和顏色
            type_config = {
                "system": {"icon": "⚙️", "color": "#6B7280"},
                "rag": {"icon": "🧠", "color": "#8B5CF6"},
                "terminal": {"icon": "💻", "color": "#10B981"},
                "error": {"icon": "❌", "color": "#EF4444"},
                "success": {"icon": "✅", "color": "#10B981"},
                "warning": {"icon": "⚠️", "color": "#F59E0B"},
                "info": {"icon": "ℹ️", "color": "#3B82F6"}
            }
            
            config = type_config.get(message_type, type_config["info"])
            
            # 創建系統訊息容器
            message_container = ctk.CTkFrame(self.system_frame, corner_radius=8)
            message_container.pack(fill="x", pady=2, padx=5)
            
            # 訊息頭部
            header_frame = ctk.CTkFrame(message_container, fg_color="transparent")
            header_frame.pack(fill="x", padx=8, pady=(5, 2))
            
            # 左側：圖標和標題
            left_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
            left_frame.pack(side="left", fill="x", expand=True)
            
            icon_title_label = ctk.CTkLabel(
                left_frame,
                text=f"{config['icon']} {title}",
                font=self.fonts['small'],
                text_color=config['color']
            )
            icon_title_label.pack(side="left")
            
            # 右側：時間戳
            time_label = ctk.CTkLabel(
                header_frame,
                text=datetime.now().strftime("%H:%M:%S"),
                font=ctk.CTkFont(size=9),
                text_color="gray"
            )
            time_label.pack(side="right")
            
            # 訊息內容（可摺疊）
            if content.strip():
                content_text = ctk.CTkTextbox(
                    message_container,
                    height=60,
                    font=ctk.CTkFont(size=10),
                    wrap="word",
                    activate_scrollbars=False
                )
                content_text.pack(fill="x", padx=8, pady=(0, 5))
                content_text.insert("1.0", content)
                content_text.configure(state="disabled")
            
            # 儲存到系統訊息列表
            message_data = {
                "type": message_type,
                "title": title,
                "content": content,
                "level": level,
                "timestamp": datetime.now(),
                "widget": message_container
            }
            self.system_messages.append(message_data)
            
            # 限制系統訊息數量（保留最新100條）
            if len(self.system_messages) > 100:
                old_message = self.system_messages.pop(0)
                old_message["widget"].destroy()
            
            # 更新統計
            self.update_system_stats()
            
            # 自動滾動到底部
            self.root.after(10, self.scroll_system_to_bottom)
            
        except Exception as e:
            print(f"添加系統訊息失敗: {e}")
    
    def scroll_system_to_bottom(self):
        """滾動系統訊息到底部"""
        try:
            self.system_frame._parent_canvas.yview_moveto(1.0)
        except:
            pass
    
    def filter_system_messages(self, filter_type):
        """過濾系統訊息"""
        try:
            for message_data in self.system_messages:
                widget = message_data["widget"]
                if filter_type == "all" or message_data["type"] == filter_type:
                    widget.pack(fill="x", pady=2, padx=5)
                else:
                    widget.pack_forget()
            
            # 更新統計
            self.update_system_stats()
            
        except Exception as e:
            print(f"過濾系統訊息失敗: {e}")
    
    def clear_system_messages(self):
        """清空系統訊息"""
        try:
            for message_data in self.system_messages:
                message_data["widget"].destroy()
            self.system_messages.clear()
            self.update_system_stats()
        except Exception as e:
            print(f"清空系統訊息失敗: {e}")
    
    def update_system_stats(self):
        """更新系統訊息統計"""
        try:
            filter_type = self.message_type_var.get()
            if filter_type == "all":
                count = len(self.system_messages)
            else:
                count = sum(1 for msg in self.system_messages if msg["type"] == filter_type)
            
            self.system_stats_label.configure(text=f"訊息數量: {count}")
        except:
            self.system_stats_label.configure(text="訊息數量: ?")
    
    def log_system_activity(self, activity_type: str, message: str, details: str = ""):
        """記錄系統活動到系統訊息"""
        self.add_system_message(activity_type, message, details)
    
    def _adjust_textbox_height(self, textbox, content):
        """動態調整文字框高度以適應內容"""
        # 估算需要的行數 - 根據消息類型調整每行字符數
        try:
            # 嘗試從父容器獲取寬度信息
            parent_frame = textbox.master
            frame_width = parent_frame.winfo_reqwidth() if hasattr(parent_frame, 'winfo_reqwidth') else 550
            # 根據實際寬度計算字符數（扣除padding和border）
            chars_per_line = max(40, (frame_width - 60) // 12)  # 12是大概的字符寬度
        except:
            chars_per_line = 65  # 默認值，適應550px寬度
        
        lines_needed = max(1, len(content) // chars_per_line + content.count('\n') + 1)
        
        # 設置高度（每行約25像素）
        height = min(lines_needed * 25, 200)  # 增加最大高度到200像素
        textbox.configure(height=height)
    
    def _fix_message_size(self, message_frame, target_width):
        """修正消息框尺寸：保持固定寬度，允許高度自適應"""
        try:
            # 暫時啟用propagation來獲取內容高度
            message_frame.pack_propagate(True)
            message_frame.update_idletasks()  # 強制更新以獲取正確尺寸
            
            # 獲取內容所需的高度
            required_height = message_frame.winfo_reqheight()
            
            # 重新設置固定寬度但使用計算的高度
            message_frame.configure(width=target_width, height=required_height)
            message_frame.pack_propagate(False)  # 現在禁用propagation來保持設定的尺寸
        except Exception as e:
            # 如果出錯，至少保持寬度固定
            message_frame.configure(width=target_width)
            message_frame.pack_propagate(False)
    
    def _recalculate_message_height(self, message_frame, sender_type):
        """重新計算並設置消息框高度"""
        try:
            target_width = 450 if sender_type == "user" else 550
            
            # 暫時啟用propagation
            message_frame.pack_propagate(True)
            message_frame.update_idletasks()
            
            # 獲取新的高度需求
            new_height = message_frame.winfo_reqheight()
            
            # 重新設置尺寸
            message_frame.configure(width=target_width, height=new_height)
            message_frame.pack_propagate(False)
            
            # 滾動到底部以顯示新內容
            self.scroll_to_bottom()
        except Exception as e:
            # 失敗時保持原狀
            pass
    
    def _add_context_menu(self, textbox):
        """為文字框添加右鍵選單"""
        context_menu = tk.Menu(self.root, tearoff=0, bg="#2B2B2B", fg="white", 
                              activebackground="#3B82F6", activeforeground="white",
                              font=self.fonts['small'])
        
        context_menu.add_command(
            label="複製 (Ctrl+C)",
            command=lambda: self._copy_selected_text(textbox)
        )
        context_menu.add_command(
            label="全選 (Ctrl+A)",
            command=lambda: self._select_all_text(textbox)
        )
        context_menu.add_separator()
        context_menu.add_command(
            label="複製整條消息",
            command=lambda: self._copy_full_message(textbox)
        )
        
        def show_context_menu(event):
            try:
                context_menu.post(event.x_root, event.y_root)
            except:
                pass
        
        # 綁定右鍵點擊事件
        textbox.bind("<Button-3>", show_context_menu)  # Windows右鍵
        textbox.bind("<Button-2>", show_context_menu)  # macOS右鍵
        
        # 綁定鍵盤快捷鍵
        textbox.bind("<Control-c>", lambda e: self._copy_selected_text(textbox))
        textbox.bind("<Control-a>", lambda e: self._select_all_text(textbox))
    
    def _copy_selected_text(self, textbox):
        """複製選中的文字"""
        try:
            selected_text = textbox.selection_get()
            if selected_text:
                self.root.clipboard_clear()
                self.root.clipboard_append(selected_text)
                print(f"📋 已複製選中文字: {selected_text[:50]}...")
        except tk.TclError:
            # 沒有選中文字時複製全部
            self._copy_full_message(textbox)
    
    def _select_all_text(self, textbox):
        """全選文字"""
        textbox.configure(state="normal")
        textbox.tag_add("sel", "1.0", "end")
        textbox.configure(state="disabled")
    
    def _copy_full_message(self, textbox):
        """複製整條消息"""
        try:
            full_text = textbox.get("1.0", "end-1c")
            self.root.clipboard_clear()
            self.root.clipboard_append(full_text)
            print(f"📋 已複製完整消息: {full_text[:50]}...")
        except Exception as e:
            print(f"❌ 複製失敗: {e}")
    
    def add_thinking_message(self):
        """添加AI思考消息，返回消息組件引用"""
        # 創建消息容器（用於控制對齊）
        message_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        message_container.pack(fill="x", pady=5)
        
        # 創建消息框架
        message_frame = ctk.CTkFrame(
            message_container,
            fg_color="#6B7280",  # 灰色表示系統消息
            corner_radius=15
        )
        # 系統消息：居中顯示
        message_frame.pack(anchor="center", padx=80, pady=0, ipadx=10, ipady=5)
        
        # 思考內容
        content_label = ctk.CTkLabel(
            message_frame,
            text="🤔 AI正在思考...",
            font=self.fonts['chat_content'],
            text_color="white",
            wraplength=600,
            justify="center",
            anchor="center"
        )
        content_label.pack(fill="x", padx=15, pady=10)
        
        # 自動滾動到底部
        self.root.after(100, self.scroll_to_bottom)
        
        return message_container
    
    def remove_thinking_message(self, thinking_widget):
        """移除AI思考消息"""
        if thinking_widget:
            thinking_widget.destroy()
    
    def _show_thinking_message(self):
        """顯示思考消息（內部方法）"""
        if not self.thinking_message_widget:
            self.thinking_message_widget = self.add_thinking_message()
    
    def _hide_thinking_message(self):
        """隱藏思考消息（內部方法）"""
        if self.thinking_message_widget:
            self.thinking_message_widget.destroy()
            self.thinking_message_widget = None
    
    def on_enter_key(self, event):
        """處理Enter鍵事件"""
        # Enter鍵發送消息
        self.send_message()
        return "break"  # 防止默認行為（插入換行）
    
    def on_shift_enter(self, event):
        """處理Shift+Enter事件"""
        # Shift+Enter插入換行，允許默認行為
        return None
    
    def send_message(self):
        """發送消息"""
        message = self.message_input.get("1.0", "end-1c").strip()
        if not message:
            return
        
        # 清空輸入框
        self.message_input.delete("1.0", "end")
        
        # 添加用戶消息
        self.add_message("user", "你", message)
        
        # 禁用發送按鈕
        self.send_button.configure(state="disabled", text="發送中...")
        
        # 異步處理AI回應
        future = self.run_async_task(self.get_ai_response(message))
        self.root.after(100, lambda: self.check_response_result(future))
    
    async def get_ai_response(self, message):
        """獲取AI回應"""
        try:
            if self.typing_switch.get():
                # 使用打字模擬
                thinking_msg_widget = None
                current_bot_message = None
                response_chunks = []
                
                async for chunk in self.core_service.generate_response_with_typing(
                    self.current_user_id, message
                ):
                    chunk_type = chunk.get("type")
                    
                    if chunk_type == "thinking":
                        # 顯示思考狀態
                        if not thinking_msg_widget:
                            thinking_msg_widget = True
                            self.root.after(0, self._show_thinking_message)
                            # 等待思考消息顯示
                            await asyncio.sleep(0.2)
                    
                    elif chunk_type == "response_start":
                        # 移除思考提示，開始顯示回應
                        if thinking_msg_widget:
                            self.root.after(0, self._hide_thinking_message)
                            thinking_msg_widget = None
                            # 等待思考消息移除
                            await asyncio.sleep(0.1)
                        
                        character_name = chunk.get("character_name", "AI")
                        self.root.after(0, lambda name=character_name: self.start_bot_message(name))
                    
                    elif chunk_type == "response_chunk":
                        # 累積回應內容
                        content = chunk.get("content", "")
                        response_chunks.append(content)
                        full_content = "".join(response_chunks)
                        self.root.after(0, lambda c=full_content: self.update_bot_message(c))
                    
                    elif chunk_type == "response_complete":
                        # 完成回應
                        conv_length = chunk.get("conversation_length", 0)
                        max_length = chunk.get("max_length", 7)
                        self.root.after(0, lambda: self.finalize_bot_message(conv_length, max_length))
                        break
            
            else:
                # 普通回應
                result = await self.core_service.generate_response(self.current_user_id, message)
                if result.get("success"):
                    character_name = result.get("character_name", "AI")
                    response = result["response"]
                    conv_length = result.get("conversation_length", 0)
                    max_length = result.get("max_length", 7)
                    
                    self.root.after(0, lambda: self.add_message("bot", character_name, response))
                    self.root.after(0, lambda: self.update_memory_display(conv_length, max_length))
                else:
                    error_msg = result.get("error", "未知錯誤")
                    self.root.after(0, lambda: self.add_message("system", "錯誤", f"發生錯誤: {error_msg}"))
        
        except Exception as e:
            self.root.after(0, lambda: self.add_message("system", "錯誤", f"發生異常: {str(e)}"))
        
        finally:
            # 重新啟用發送按鈕
            self.root.after(0, lambda: self.send_button.configure(state="normal", text="發送"))
    
    def start_bot_message(self, character_name):
        """開始顯示bot消息"""
        # 創建消息容器（用於控制對齊）- 添加 expand=True
        message_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        message_container.pack(fill="x", pady=5, expand=True)
        
        # 創建一個空的bot消息框架，稍後更新內容
        message_frame = ctk.CTkFrame(
            message_container,
            fg_color="#8B5CF6",
            corner_radius=15
        )
        # AI消息：左對齊，固定寬度但允許高度自適應 - 比用戶消息稍寬一些
        message_frame.configure(width=550)  # 明確設定寬度，比用戶消息寬100px
        message_frame.pack(side="left", padx=(20, 50), pady=0, ipadx=10, ipady=5)
        # 使用 winfo_reqwidth() 來保持寬度但允許高度變化
        self.root.after(1, lambda: self._fix_message_size(message_frame, 550))
        
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
        
        # 內容區域 - 使用 CTkTextbox 支持文字選擇
        self.current_content_textbox = ctk.CTkTextbox(
            message_frame,
            font=self.fonts['chat_content'],
            text_color="white",
            fg_color="transparent",
            corner_radius=0,
            border_width=0,
            wrap="word",
            height=30,  # 初始高度
            activate_scrollbars=False
        )
        self.current_content_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # 設置為可編輯（為了動態更新內容）
        self.current_content_textbox.configure(state="normal")
        
        # 添加右鍵選單
        self._add_context_menu(self.current_content_textbox)
        
        self.chat_messages.append(message_container)
        self.current_typing_message = message_frame
        self.current_typing_indicator = typing_indicator
        self.scroll_to_bottom()
    
    def update_bot_message(self, content):
        """更新bot消息內容"""
        if hasattr(self, 'current_content_textbox'):
            # 清空現有內容並插入新內容
            self.current_content_textbox.delete("1.0", "end")
            self.current_content_textbox.insert("1.0", content)
            
            # 動態調整高度
            self._adjust_textbox_height(self.current_content_textbox, content)
            
            # 重新計算消息框高度
            if hasattr(self, 'current_typing_message'):
                self.root.after(10, lambda: self._recalculate_message_height(self.current_typing_message, "bot"))
            
            self.scroll_to_bottom()
    
    def finalize_bot_message(self, conv_length, max_length):
        """完成bot消息"""
        if hasattr(self, 'current_typing_indicator'):
            # 更新時間戳，移除打字指示器
            self.current_typing_indicator.configure(text=datetime.now().strftime("%H:%M:%S"))
        
        # 將textbox設為只讀
        if hasattr(self, 'current_content_textbox'):
            self.current_content_textbox.configure(state="disabled")
        
        self.current_typing_message = None
        self.current_content_textbox = None
        self.update_memory_display(conv_length, max_length)
    
    def clear_chat(self):
        """清空聊天"""
        for message_container in self.chat_messages:
            message_container.destroy()
        self.chat_messages.clear()
    
    def check_response_result(self, future):
        """檢查回應結果"""
        if future.done():
            try:
                future.result()
            except Exception as e:
                self.add_message("system", "錯誤", f"處理失敗: {str(e)}")
                self.send_button.configure(state="normal", text="發送")
        else:
            self.root.after(100, lambda: self.check_response_result(future))
    
    # ==================== 控制功能 ====================
    
    def quick_toggle_rag(self):
        """快速切換RAG"""
        enabled = self.rag_switch.get()
        self.rag_detail_switch.configure(state="normal" if enabled else "disabled")
        if enabled:
            self.rag_detail_switch.select()
        else:
            self.rag_detail_switch.deselect()
        self.toggle_rag()
    
    def quick_toggle_typing(self):
        """快速切換打字模擬"""
        enabled = self.typing_switch.get()
        self.typing_detail_switch.configure(state="normal" if enabled else "disabled")
        if enabled:
            self.typing_detail_switch.select()
        else:
            self.typing_detail_switch.deselect()
        self.toggle_typing()
    
    def quick_toggle_traditional(self):
        """快速切換簡繁轉換"""
        enabled = self.s2t_switch.get()
        if hasattr(self, 'traditional_switch'):
            self.traditional_switch.configure(state="normal" if enabled else "disabled")
            if enabled:
                self.traditional_switch.select()
            else:
                self.traditional_switch.deselect()
        self.toggle_traditional_chinese_direct(enabled)
    
    def toggle_rag(self):
        """切換RAG狀態"""
        enabled = self.rag_detail_switch.get()
        result = self.core_service.toggle_rag(enabled)
        message = result.get("message", "RAG狀態已更新")
        self.update_status(message)
        
        # 記錄到系統訊息
        status = "啟用" if enabled else "禁用"
        self.log_system_activity("rag", f"RAG檢索{status}", message)
        
        # 同步頭部開關
        if enabled != self.rag_switch.get():
            if enabled:
                self.rag_switch.select()
            else:
                self.rag_switch.deselect()
    
    def toggle_typing(self):
        """切換打字模擬"""
        enabled = self.typing_detail_switch.get()
        result = self.core_service.toggle_typing_simulation(enabled)
        message = result.get("message", "打字模擬狀態已更新")
        self.update_status(message)
        
        # 記錄到系統訊息
        status = "啟用" if enabled else "禁用"
        self.log_system_activity("system", f"打字模擬{status}", message)
        
        # 同步頭部開關
        if enabled != self.typing_switch.get():
            if enabled:
                self.typing_switch.select()
            else:
                self.typing_switch.deselect()
    
    def toggle_line_break(self):
        """切換智慧換行"""
        enabled = self.line_break_switch.get()
        result = self.core_service.toggle_line_break(enabled)
        self.update_status(result.get("message", "智慧換行狀態已更新"))
    
    def toggle_traditional_chinese(self):
        """切換簡繁轉換"""
        enabled = self.traditional_switch.get()
        result = self.core_service.toggle_traditional_chinese(enabled)
        if result.get("success"):
            self.update_status(result.get("message", "簡繁轉換狀態已更新"))
            # 更新狀態顯示
            status_text = "已啟用" if enabled else "已禁用"
            self.conversion_status_label.configure(text=f"狀態: {status_text}")
            
            # 同步快速開關
            if enabled != self.s2t_switch.get():
                if enabled:
                    self.s2t_switch.select()
                else:
                    self.s2t_switch.deselect()
        else:
            self.update_status(f"設置失敗: {result.get('error', '未知錯誤')}")
            # 如果設置失敗，回復開關狀態
            if enabled:
                self.traditional_switch.deselect()
            else:
                self.traditional_switch.select()
    
    def toggle_traditional_chinese_direct(self, enabled):
        """直接切換簡繁轉換（用於快速開關）"""
        result = self.core_service.toggle_traditional_chinese(enabled)
        if result.get("success"):
            self.update_status(result.get("message", "簡繁轉換狀態已更新"))
            # 更新詳細設置頁面的狀態顯示
            if hasattr(self, 'conversion_status_label'):
                status_text = "已啟用" if enabled else "已禁用"
                self.conversion_status_label.configure(text=f"狀態: {status_text}")
        else:
            self.update_status(f"設置失敗: {result.get('error', '未知錯誤')}")
            # 如果設置失敗，回復開關狀態
            if enabled:
                self.s2t_switch.deselect()
            else:
                self.s2t_switch.select()
    
    def on_typing_preset_change(self, preset):
        """打字速度預設變更"""
        result = self.core_service.set_typing_preset(preset)
        if result.get("success"):
            self.update_status(result.get("message", f"已套用預設: {preset}"))
        else:
            self.update_status(f"設置失敗: {result.get('error', '未知錯誤')}")
    
    def clear_memory(self):
        """清除記憶"""
        result = self.core_service.clear_user_memory(self.current_user_id)
        if result.get("success"):
            self.update_status(result.get("message", "記憶已清除"))
            self.update_memory_display()
        else:
            messagebox.showerror("錯誤", result.get("error", "清除失敗"))
    
    # ==================== RAG 功能 ====================
    
    def upload_document(self):
        """上傳文檔"""
        file_path = filedialog.askopenfilename(
            title="選擇文檔",
            filetypes=[
                ("文本文件", "*.txt"),
                ("PDF文件", "*.pdf"),
                ("Word文件", "*.docx"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            filename = Path(file_path).name
            self.update_status("正在上傳文檔...")
            self.log_system_activity("rag", "開始上傳文檔", f"文件: {filename}")
            
            future = self.run_async_task(self.core_service.add_document(file_path))
            self.root.after(100, lambda: self.check_upload_result(future, filename))
    
    def check_upload_result(self, future, filename):
        """檢查上傳結果"""
        if future.done():
            try:
                result = future.result()
                if result.get("success"):
                    message = f"文檔 {filename} 上傳成功"
                    self.update_status(message)
                    self.log_system_activity("success", "文檔上傳成功", filename)
                    self.refresh_stats()
                else:
                    error_msg = result.get("error", "未知錯誤")
                    self.log_system_activity("error", "文檔上傳失敗", f"{filename}: {error_msg}")
                    messagebox.showerror("上傳失敗", error_msg)
            except Exception as e:
                self.log_system_activity("error", "文檔上傳異常", f"{filename}: {str(e)}")
                messagebox.showerror("上傳異常", str(e))
        else:
            self.root.after(100, lambda: self.check_upload_result(future, filename))
    
    def search_knowledge(self):
        """搜索知識庫"""
        query = self.search_entry.get().strip()
        if not query:
            return
        
        future = self.run_async_task(self.core_service.search_knowledge_base(query))
        self.root.after(100, lambda: self.check_search_result(future))
    
    def check_search_result(self, future):
        """檢查搜索結果"""
        if future.done():
            try:
                result = future.result()
                if result.get("success"):
                    self.display_search_results(result["results"])
                else:
                    self.search_result.delete("1.0", "end")
                    self.search_result.insert("1.0", f"搜索失敗: {result.get('error', '未知錯誤')}")
            except Exception as e:
                self.search_result.delete("1.0", "end")
                self.search_result.insert("1.0", f"搜索異常: {str(e)}")
        else:
            self.root.after(100, lambda: self.check_search_result(future))
    
    def display_search_results(self, results):
        """顯示搜索結果"""
        self.search_result.delete("1.0", "end")
        
        if not results:
            self.search_result.insert("1.0", "未找到相關內容")
            return
        
        content = ""
        for i, result in enumerate(results, 1):
            content += f"結果 {i}:\n"
            content += f"相似度: {result.get('similarity', 0):.3f}\n"
            content += f"內容: {result.get('content', '')[:200]}...\n"
            content += "-" * 50 + "\n\n"
        
        self.search_result.insert("1.0", content)
    
    def clear_knowledge_base(self):
        """清空知識庫"""
        # 確認對話框
        result = messagebox.askyesno("確認", "確定要清空整個知識庫嗎？此操作不可撤銷。")
        if result:
            future = self.run_async_task(self.core_service.clear_knowledge_base())
            self.root.after(100, lambda: self.check_clear_result(future))
    
    def check_clear_result(self, future):
        """檢查清空結果"""
        if future.done():
            try:
                result = future.result()
                if result.get("success"):
                    self.update_status("知識庫已清空")
                    self.refresh_stats()
                else:
                    messagebox.showerror("清空失敗", result.get("error", "未知錯誤"))
            except Exception as e:
                messagebox.showerror("清空異常", str(e))
        else:
            self.root.after(100, lambda: self.check_clear_result(future))
    
    # ==================== 系統功能 ====================
    
    def check_service_status(self):
        """檢查服務狀態"""
        try:
            if self.core_service._initialized:
                self.character_status_label.configure(text="🟢 已連接")
                self.character_name_label.configure(text=self.core_service.character_name)
                self.update_memory_display()
                
                # 記錄服務連接狀態
                if not hasattr(self, '_service_connected_logged'):
                    self.log_system_activity("success", "服務已連接", f"角色: {self.core_service.character_name}")
                    self._service_connected_logged = True
                
                # 檢查簡繁轉換狀態
                self.check_traditional_chinese_status()
            else:
                self.character_status_label.configure(text="🔴 未連接")
                if not hasattr(self, '_service_disconnected_logged'):
                    self.log_system_activity("warning", "服務未連接", "正在等待核心服務初始化...")
                    self._service_disconnected_logged = True
        except Exception as e:
            self.character_status_label.configure(text="❌ 錯誤")
            self.log_system_activity("error", "服務狀態檢查失敗", str(e))
        
        # 5秒後再次檢查
        self.root.after(5000, self.check_service_status)
    
    def check_traditional_chinese_status(self):
        """檢查簡繁轉換狀態"""
        try:
            status = self.core_service.get_traditional_chinese_status()
            if status.get("success"):
                enabled = status.get("conversion_enabled", False)
                available = status.get("opencc_available", False)
                initialized = status.get("converter_initialized", False)
                
                # 更新詳細設置頁面的開關狀態
                if hasattr(self, 'traditional_switch'):
                    if enabled:
                        self.traditional_switch.select()
                    else:
                        self.traditional_switch.deselect()
                
                # 更新快速開關狀態
                if enabled:
                    self.s2t_switch.select()
                else:
                    self.s2t_switch.deselect()
                
                # 更新狀態顯示
                if not available:
                    status_text = "OpenCC 未安裝"
                    if hasattr(self, 'traditional_switch'):
                        self.traditional_switch.configure(state="disabled")
                    self.s2t_switch.configure(state="disabled")
                elif not initialized:
                    status_text = "轉換器未初始化"
                    if hasattr(self, 'traditional_switch'):
                        self.traditional_switch.configure(state="disabled")
                    self.s2t_switch.configure(state="disabled")
                elif enabled:
                    status_text = "已啟用"
                    if hasattr(self, 'traditional_switch'):
                        self.traditional_switch.configure(state="normal")
                    self.s2t_switch.configure(state="normal")
                else:
                    status_text = "已禁用"
                    if hasattr(self, 'traditional_switch'):
                        self.traditional_switch.configure(state="normal")
                    self.s2t_switch.configure(state="normal")
                
                if hasattr(self, 'conversion_status_label'):
                    self.conversion_status_label.configure(text=f"狀態: {status_text}")
            else:
                if hasattr(self, 'conversion_status_label'):
                    self.conversion_status_label.configure(text="狀態: 檢查失敗")
                if hasattr(self, 'traditional_switch'):
                    self.traditional_switch.configure(state="disabled")
                self.s2t_switch.configure(state="disabled")
        except Exception as e:
            if hasattr(self, 'conversion_status_label'):
                self.conversion_status_label.configure(text=f"狀態: 錯誤 - {str(e)[:20]}")
            if hasattr(self, 'traditional_switch'):
                self.traditional_switch.configure(state="disabled")
            self.s2t_switch.configure(state="disabled")
    
    def update_memory_display(self, conv_length=None, max_length=None):
        """更新記憶顯示"""
        try:
            if conv_length is None:
                result = self.core_service.get_user_memory_status(self.current_user_id)
                if result.get("success"):
                    conv_length = result.get("memory_count", 0)
                    max_length = result.get("max_length", 7)
                else:
                    conv_length, max_length = 0, 7
            
            self.memory_status_label.configure(text=f"記憶: {conv_length}/{max_length}")
        except Exception:
            self.memory_status_label.configure(text="記憶: ?/?")
    
    def refresh_stats(self):
        """刷新統計信息"""
        try:
            stats = self.core_service.get_stats()
            if stats.get("success"):
                doc_count = stats.get("total_documents", 0)
                self.doc_count_label.configure(text=f"文檔數量: {doc_count}")
                self.update_status("統計信息已刷新")
        except Exception as e:
            self.update_status(f"刷新統計失敗: {str(e)}")
    
    def refresh_system_info(self):
        """刷新系統信息"""
        try:
            stats = self.core_service.get_stats()
            if stats.get("success"):
                info_text = json.dumps(stats, ensure_ascii=False, indent=2)
                self.system_info.delete("1.0", "end")
                self.system_info.insert("1.0", info_text)
        except Exception as e:
            self.system_info.delete("1.0", "end")
            self.system_info.insert("1.0", f"獲取系統信息失敗: {str(e)}")
    
    def show_model_info(self):
        """顯示模型信息"""
        try:
            model_info = self.core_service.get_model_info()
            info_text = json.dumps(model_info, ensure_ascii=False, indent=2)
            
            # 創建新窗口
            info_window = ctk.CTkToplevel(self.root)
            info_window.title("模型信息")
            info_window.geometry("600x400")
            
            text_widget = ctk.CTkTextbox(info_window, font=self.fonts['monospace'])
            text_widget.pack(fill="both", expand=True, padx=20, pady=20)
            text_widget.insert("1.0", info_text)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"獲取模型信息失敗: {str(e)}")
    
    def show_conversion_info(self):
        """顯示簡繁轉換詳細信息"""
        try:
            status = self.core_service.get_traditional_chinese_status()
            
            # 格式化信息
            info_lines = [
                "=== 簡繁轉換狀態 ===\n",
                f"OpenCC 可用性: {'✅ 已安裝' if status.get('opencc_available') else '❌ 未安裝'}",
                f"轉換器初始化: {'✅ 已初始化' if status.get('converter_initialized') else '❌ 未初始化'}",
                f"轉換功能: {'✅ 已啟用' if status.get('conversion_enabled') else '❌ 已禁用'}",
            ]
            
            # 添加配置文件信息
            if status.get('config_file'):
                info_lines.append(f"配置文件: {status['config_file']}")
            
            # 添加測試結果
            test_result = status.get('test_result')
            if test_result:
                info_lines.extend([
                    "\n=== 轉換測試 ===",
                    f"原文: {test_result['original']}",
                    f"轉換: {test_result['converted']}"
                ])
            
            # 添加智慧換行統計
            line_break_stats = self.core_service.get_line_break_stats()
            if line_break_stats.get('success'):
                stats = line_break_stats['stats']
                info_lines.extend([
                    "\n=== 智慧換行統計 ===",
                    f"處理次數: {stats.get('total_count', 0)}",
                    f"修改次數: {stats.get('modified_count', 0)}",
                    f"錯誤次數: {stats.get('error_count', 0)}",
                    f"啟用狀態: {'✅ 已啟用' if line_break_stats.get('enabled') else '❌ 已禁用'}"
                ])
            
            info_text = "\n".join(info_lines)
            
            # 創建新窗口顯示信息
            info_window = ctk.CTkToplevel(self.root)
            info_window.title("過濾器詳細信息")
            info_window.geometry("500x600")
            
            text_widget = ctk.CTkTextbox(
                info_window, 
                font=self.fonts['monospace']
            )
            text_widget.pack(fill="both", expand=True, padx=20, pady=20)
            text_widget.insert("1.0", info_text)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"獲取轉換信息失敗: {str(e)}")
    
    def export_chat_log(self):
        """導出聊天記錄"""
        try:
            if not self.chat_messages:
                messagebox.showinfo("提示", "沒有聊天記錄可導出")
                return
            
            file_path = filedialog.asksaveasfilename(
                title="保存聊天記錄",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
            )
            
            if file_path:
                # 簡化的聊天記錄導出
                chat_content = f"VTuber AI 聊天記錄\n導出時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n\n"
                chat_content += f"總計 {len(self.chat_messages)} 條消息\n\n"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(chat_content)
                
                self.update_status(f"聊天記錄已保存")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"導出失敗: {str(e)}")
    
    # ==================== 輔助功能 ====================
    
    def update_status(self, message):
        """更新狀態"""
        self.status_label.configure(text=message)
        # 3秒後恢復默認狀態
        self.root.after(3000, lambda: self.status_label.configure(text="就緒"))
    
    def update_time(self):
        """更新時間"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.root.after(1000, self.update_time)
    
    def run(self):
        """運行GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def show_welcome_messages(self):
        """顯示歡迎訊息"""
        self.log_system_activity("info", "系統啟動", "VTuber AI 助手已啟動")
        self.log_system_activity("system", "版面配置", "使用三分割布局：聊天區 + 系統訊息 + 控制面板")
        self.log_system_activity("info", "功能說明", "系統訊息支援類型過濾：system, rag, terminal, error, success")
    
    def on_closing(self):
        """關閉處理"""
        try:
            self.log_system_activity("warning", "系統關閉", "VTuber AI 助手正在關閉...")
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.root.destroy()
        except:
            pass


if __name__ == "__main__":
    print("請使用 gui_launcher.py 啟動GUI")
