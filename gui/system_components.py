"""
系統訊息和狀態組件
"""
import customtkinter as ctk
from datetime import datetime
from typing import List, Dict, Any

class SystemMessagePanel:
    """系統訊息面板"""
    
    def __init__(self, parent, fonts):
        self.parent = parent
        self.fonts = fonts
        self.system_messages: List[Dict] = []
        self.setup_ui()
    
    def setup_ui(self):
        """設置系統訊息面板UI"""
        # 中間系統訊息區域 - 減少寬度給聊天面板更多空間
        self.system_container = ctk.CTkFrame(self.parent, width=280)  # 從300減少到280
        self.system_container.pack(side="left", fill="y", padx=(5, 5), pady=20)
        self.system_container.pack_propagate(False)
        
        # 標題
        system_header = ctk.CTkLabel(
            self.system_container,
            text="📋 系統訊息",
            font=self.fonts['subtitle']
        )
        system_header.pack(pady=(10, 5))
        
        # 過濾器控制
        self._create_filter_controls()
        
        # 系統訊息滾動區域
        self.system_frame = ctk.CTkScrollableFrame(
            self.system_container, 
            corner_radius=10,
            label_text="最新訊息"
        )
        self.system_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # 統計標籤
        self.system_stats_label = ctk.CTkLabel(
            self.system_container,
            text="訊息數量: 0",
            font=self.fonts['small']
        )
        self.system_stats_label.pack(pady=(0, 10))
    
    def _create_filter_controls(self):
        """創建過濾器控制"""
        type_frame = ctk.CTkFrame(self.system_container, fg_color="transparent")
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
    
    def add_system_message(self, message_type: str, title: str, content: str, level: str = "info"):
        """添加系統訊息"""
        try:
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
            
            # 訊息內容
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
            
            # 儲存到列表
            message_data = {
                "type": message_type,
                "title": title,
                "content": content,
                "level": level,
                "timestamp": datetime.now(),
                "widget": message_container
            }
            self.system_messages.append(message_data)
            
            # 限制數量
            if len(self.system_messages) > 100:
                old_message = self.system_messages.pop(0)
                old_message["widget"].destroy()
            
            self.update_system_stats()
            self.parent.after(10, self.scroll_system_to_bottom)
            
        except Exception as e:
            print(f"添加系統訊息失敗: {e}")
    
    def filter_system_messages(self, filter_type):
        """過濾系統訊息"""
        try:
            for message_data in self.system_messages:
                widget = message_data["widget"]
                if filter_type == "all" or message_data["type"] == filter_type:
                    widget.pack(fill="x", pady=2, padx=5)
                else:
                    widget.pack_forget()
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
        """更新統計"""
        try:
            filter_type = self.message_type_var.get()
            if filter_type == "all":
                count = len(self.system_messages)
            else:
                count = sum(1 for msg in self.system_messages if msg["type"] == filter_type)
            self.system_stats_label.configure(text=f"訊息數量: {count}")
        except:
            self.system_stats_label.configure(text="訊息數量: ?")
    
    def scroll_system_to_bottom(self):
        """滾動到底部"""
        try:
            self.system_frame._parent_canvas.yview_moveto(1.0)
        except:
            pass

class HeaderPanel:
    """頂部資訊面板"""
    
    def __init__(self, parent, fonts, on_quick_toggle_callbacks):
        self.parent = parent
        self.fonts = fonts
        self.callbacks = on_quick_toggle_callbacks
        self.setup_ui()
    
    def setup_ui(self):
        """設置頂部面板UI"""
        header_frame = ctk.CTkFrame(self.parent, height=80, corner_radius=15)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        header_frame.pack_propagate(False)
        
        # 角色信息
        self._create_character_info(header_frame)
        
        # 快速控制
        self._create_quick_controls(header_frame)
    
    def _create_character_info(self, header_frame):
        """創建角色信息"""
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
    
    def _create_quick_controls(self, header_frame):
        """創建快速控制"""
        control_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        control_frame.pack(side="right", padx=20, pady=10)
        
        # RAG 開關
        self.rag_switch = ctk.CTkSwitch(
            control_frame,
            text="RAG檢索",
            command=self.callbacks.get('rag'),
            font=self.fonts['small']
        )
        self.rag_switch.pack(side="top", pady=2)
        self.rag_switch.select()
        
        # 打字模擬開關
        self.typing_switch = ctk.CTkSwitch(
            control_frame,
            text="打字模擬",
            command=self.callbacks.get('typing'),
            font=self.fonts['small']
        )
        self.typing_switch.pack(side="top", pady=2)
        self.typing_switch.select()
        
        # 簡繁轉換開關
        self.s2t_switch = ctk.CTkSwitch(
            control_frame,
            text="簡繁轉換",
            command=self.callbacks.get('s2t'),
            font=self.fonts['small']
        )
        self.s2t_switch.pack(side="top", pady=2)
        self.s2t_switch.select()
    
    def update_character_info(self, name: str, status: str):
        """更新角色信息"""
        self.character_name_label.configure(text=name)
        self.character_status_label.configure(text=status)

class StatusBar:
    """狀態欄"""
    
    def __init__(self, parent, fonts):
        self.parent = parent
        self.fonts = fonts
        self.setup_ui()
    
    def setup_ui(self):
        """設置狀態欄UI"""
        status_frame = ctk.CTkFrame(self.parent, height=40, corner_radius=10)
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
        
        self.update_time()
    
    def update_status(self, message: str):
        """更新狀態"""
        self.status_label.configure(text=message)
        self.parent.after(3000, lambda: self.status_label.configure(text="就緒"))
    
    def update_time(self):
        """更新時間"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.parent.after(1000, self.update_time)