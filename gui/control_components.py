"""
控制面板組件
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Dict, Any, Callable
import json

class ControlPanel:
    """控制面板組件"""
    
    def __init__(self, parent, fonts, event_handlers: Dict[str, Callable]):
        self.parent = parent
        self.fonts = fonts
        self.handlers = event_handlers
        self.setup_ui()
    
    def setup_ui(self):
        """設置控制面板UI"""
        # 右側控制面板 - 減少寬度給聊天面板更多空間
        control_frame = ctk.CTkFrame(self.parent, width=240, corner_radius=15)  # 從250減少到240
        control_frame.pack(side="right", fill="y", padx=(0, 20), pady=20)
        control_frame.pack_propagate(False)
        
        # 標題
        title_label = ctk.CTkLabel(
            control_frame,
            text="🎛️ 控制面板",
            font=self.fonts['subtitle']
        )
        title_label.pack(pady=(20, 10))
        
        # 分頁控制
        self.tabview = ctk.CTkTabview(control_frame, width=220, height=580)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)
        
        # 創建各個分頁
        self._setup_ai_tab()
        self._setup_rag_tab()
        self._setup_system_tab()
    
    def _setup_ai_tab(self):
        """AI設置分頁"""
        ai_tab = self.tabview.add("AI設置")
        
        # RAG控制區域
        self._create_rag_section(ai_tab)
        
        # 打字模擬控制
        self._create_typing_section(ai_tab)
        
        # 智慧換行控制
        self._create_line_break_section(ai_tab)
        
        # 簡繁轉換控制
        self._create_conversion_section(ai_tab)
        
        # 記憶管理
        self._create_memory_section(ai_tab)
    
    def _create_rag_section(self, parent):
        """創建RAG控制區域"""
        rag_frame = ctk.CTkFrame(parent, corner_radius=10)
        rag_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            rag_frame,
            text="🧠 智能檢索",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.rag_detail_switch = ctk.CTkSwitch(
            rag_frame,
            text="啟用 RAG 檢索",
            command=self.handlers.get('toggle_rag')
        )
        self.rag_detail_switch.pack(pady=(10, 15))
        self.rag_detail_switch.select()
    
    def _create_typing_section(self, parent):
        """創建打字模擬區域"""
        typing_frame = ctk.CTkFrame(parent, corner_radius=10)
        typing_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            typing_frame,
            text="⌨️ 打字模擬",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.typing_detail_switch = ctk.CTkSwitch(
            typing_frame,
            text="啟用打字效果",
            command=self.handlers.get('toggle_typing')
        )
        self.typing_detail_switch.pack(pady=5)
        self.typing_detail_switch.select()
        
        # 速度預設
        ctk.CTkLabel(typing_frame, text="速度預設:").pack(pady=(10, 0))
        self.typing_preset = ctk.CTkOptionMenu(
            typing_frame,
            values=["slow", "normal", "fast", "very_fast", "thoughtful"],
            command=self.handlers.get('typing_preset_change')
        )
        self.typing_preset.pack(pady=(5, 15))
        self.typing_preset.set("normal")
    
    def _create_line_break_section(self, parent):
        """創建智慧換行區域"""
        line_break_frame = ctk.CTkFrame(parent, corner_radius=10)
        line_break_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            line_break_frame,
            text="📝 智慧換行",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.line_break_switch = ctk.CTkSwitch(
            line_break_frame,
            text="啟用智慧換行",
            command=self.handlers.get('toggle_line_break')
        )
        self.line_break_switch.pack(pady=(5, 15))
        self.line_break_switch.select()
    
    def _create_conversion_section(self, parent):
        """創建簡繁轉換區域"""
        conversion_frame = ctk.CTkFrame(parent, corner_radius=10)
        conversion_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            conversion_frame,
            text="🔄 簡繁轉換",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 5))
        
        self.traditional_switch = ctk.CTkSwitch(
            conversion_frame,
            text="啟用簡繁轉換",
            command=self.handlers.get('toggle_traditional_chinese')
        )
        self.traditional_switch.pack(pady=5)
        self.traditional_switch.select()
        
        self.conversion_status_label = ctk.CTkLabel(
            conversion_frame,
            text="狀態: 檢查中...",
            font=self.fonts['small']
        )
        self.conversion_status_label.pack(pady=(5, 15))
    
    def _create_memory_section(self, parent):
        """創建記憶管理區域"""
        memory_frame = ctk.CTkFrame(parent, corner_radius=10)
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
            command=self.handlers.get('clear_memory'),
            fg_color="darkred",
            hover_color="red"
        ).pack(pady=(5, 15))
    
    def _setup_rag_tab(self):
        """知識庫分頁"""
        rag_tab = self.tabview.add("知識庫")
        
        # 文檔管理
        self._create_document_section(rag_tab)
        
        # 搜索測試
        self._create_search_section(rag_tab)
        
        # 統計信息
        self._create_stats_section(rag_tab)
    
    def _create_document_section(self, parent):
        """創建文檔管理區域"""
        doc_frame = ctk.CTkFrame(parent, corner_radius=10)
        doc_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            doc_frame,
            text="📚 文檔管理",
            font=self.fonts['subtitle']
        ).pack(pady=(15, 10))
        
        ctk.CTkButton(
            doc_frame,
            text="📁 上傳文檔",
            command=self.handlers.get('upload_document')
        ).pack(fill="x", padx=15, pady=5)
        
        ctk.CTkButton(
            doc_frame,
            text="🗑️ 清空知識庫",
            command=self.handlers.get('clear_knowledge_base'),
            fg_color="darkred",
            hover_color="red"
        ).pack(fill="x", padx=15, pady=(5, 15))
    
    def _create_search_section(self, parent):
        """創建搜索測試區域"""
        search_frame = ctk.CTkFrame(parent, corner_radius=10)
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
            command=self.handlers.get('search_knowledge')
        ).pack(fill="x", padx=15, pady=(5, 10))
        
        self.search_result = ctk.CTkTextbox(
            search_frame,
            height=150,
            font=self.fonts['small']
        )
        self.search_result.pack(fill="both", expand=True, padx=15, pady=(0, 15))
    
    def _create_stats_section(self, parent):
        """創建統計信息區域"""
        stats_frame = ctk.CTkFrame(parent, corner_radius=10)
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
            command=self.handlers.get('refresh_stats')
        ).pack(fill="x", padx=15, pady=(5, 15))
    
    def _setup_system_tab(self):
        """系統分頁"""
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
        
        buttons = [
            ("刷新信息", self.handlers.get('refresh_system_info')),
            ("模型信息", self.handlers.get('show_model_info')),
            ("轉換狀態", self.handlers.get('show_conversion_info')),
            ("導出日誌", self.handlers.get('export_chat_log'))
        ]
        
        for i, (text, command) in enumerate(buttons):
            side = "left" if i < 3 else "right"
            padx = (0, 5) if i < 2 else (5, 0) if i == 2 else 0
            
            ctk.CTkButton(
                button_frame,
                text=text,
                command=command,
                width=80
            ).pack(side=side, padx=padx)