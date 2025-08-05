#!/usr/bin/env python3
"""
VTuber AI LLM 終端版本 - 輕量級前端
使用核心服務層處理所有AI邏輯
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import yaml
import colorama
from colorama import Fore, Style, Back

from src.core_service import VTuberCoreService
from src.utils.logger import setup_logger

colorama.init()

def load_config(config_path: str = "config.yaml") -> dict:
    """載入配置文件，並將所有相對路徑轉為以專案根目錄為基準的絕對路徑"""
    try:
        config_path = Path(config_path)
        # 專案根目錄 = main.py 的同一層（與discord_bot.py保持一致）
        project_root = Path(__file__).parent.parent.parent.resolve()
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 以專案根目錄為基準處理路徑
        config['models']['llm']['model_path'] = str((project_root / config['models']['llm']['model_path']).resolve())
        config['models']['embedding']['model_path'] = str((project_root / config['models']['embedding']['model_path']).resolve())
        config['rag']['vector_db']['persist_directory'] = str((project_root / config['rag']['vector_db']['persist_directory']).resolve())
        config['system']['cache_dir'] = str((project_root / config['system']['cache_dir']).resolve())
        config['system']['log_dir'] = str((project_root / config['system']['log_dir']).resolve())
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件 {config_path} 未找到")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ 配置文件解析錯誤: {e}")
        sys.exit(1)

    


class VTuberTerminal:
    """輕量級終端界面"""
    
    def __init__(self, config: dict):
        self.config = config
        self.core_service = VTuberCoreService(config)
        self.running = True
        self.user_id = "terminal_user"  # 終端用戶ID
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """初始化"""
        print(f"\n{Fore.CYAN}🚀 正在初始化 VTuber AI LLM 系統...{Style.RESET_ALL}")
        
        success = await self.core_service.initialize()
        if not success:
            print(f"{Fore.RED}❌ 初始化失敗{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.GREEN}✅ 系統初始化完成!{Style.RESET_ALL}\n")
        return True
    
    def print_welcome(self):
        """顯示歡迎信息"""
        stats = self.core_service.get_stats()
        character_name = stats.get('character_name', 'AI助手')
        character_personality = stats.get('character_personality', '智能助手')
        
        print(f"\n{Back.MAGENTA}{Fore.WHITE} VTuber AI LLM 終端版本 {Style.RESET_ALL}")
        print(f"{Fore.CYAN}═══════════════════════════════════════{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🎭 角色: {character_name}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💫 性格: {character_personality}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}═══════════════════════════════════════{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}可用指令:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}/help{Style.RESET_ALL}     - 顯示幫助信息")
        print(f"  {Fore.CYAN}/add{Style.RESET_ALL}      - 添加文檔到知識庫")
        print(f"  {Fore.CYAN}/search{Style.RESET_ALL}   - 搜索知識庫")
        print(f"  {Fore.CYAN}/stats{Style.RESET_ALL}    - 顯示系統統計")
        print(f"  {Fore.CYAN}/model{Style.RESET_ALL}    - 顯示模型信息")
        print(f"  {Fore.CYAN}/clear{Style.RESET_ALL}    - 清空知識庫")
        print(f"  {Fore.CYAN}/rag on{Style.RESET_ALL}  - 啟用RAG檢索")
        print(f"  {Fore.CYAN}/rag off{Style.RESET_ALL} - 禁用RAG檢索")
        print(f"  {Fore.CYAN}/memory{Style.RESET_ALL}   - 查看對話記憶")
        print(f"  {Fore.CYAN}/clear_memory{Style.RESET_ALL} - 清除對話記憶")
        print(f"  {Fore.CYAN}/s2t on/off{Style.RESET_ALL} - 控制簡繁轉換")
        print(f"  {Fore.CYAN}/s2t status{Style.RESET_ALL} - 查看轉換狀態")
        print(f"  {Fore.CYAN}/typing on/off{Style.RESET_ALL} - 控制打字模擬")
        print(f"  {Fore.CYAN}/exit{Style.RESET_ALL}     - 退出程序")
        print(f"\n{Fore.MAGENTA}💬 直接輸入消息開始對話!{Style.RESET_ALL}\n")
    
    async def handle_command(self, user_input: str):
        """處理用戶指令"""
        command = user_input.strip().lower()
        
        if command == "/help":
            self.print_help()
        elif command == "/exit":
            print(f"{Fore.YELLOW}👋 再見! VTuber AI LLM 系統即將關閉...{Style.RESET_ALL}")
            self.running = False
        elif command == "/stats":
            await self.show_stats()
        elif command == "/model":
            await self.show_model_info()
        elif command == "/clear":
            await self.clear_knowledge_base()
        elif command == "/rag on":
            result = self.core_service.toggle_rag(True)
            print(f"{Fore.GREEN}✅ {result['message']}{Style.RESET_ALL}")
        elif command == "/rag off":
            result = self.core_service.toggle_rag(False)
            print(f"{Fore.YELLOW}⚠️ {result['message']}{Style.RESET_ALL}")
        elif command.startswith("/add "):
            file_path = command[5:].strip()
            await self.add_document(file_path)
        elif command.startswith("/search "):
            query = command[8:].strip()
            await self.search_knowledge_base(query)
        elif command == "/memory":
            await self.show_conversation_memory()
        elif command == "/clear_memory":
            await self.clear_conversation_memory()
        elif command.startswith("/s2t"):
            await self.handle_traditional_chinese_command(command)
        elif command.startswith("/typing"):
            await self.handle_typing_command(command)
        else:
            print(f"{Fore.RED}❌ 未知指令: {command}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}輸入 /help 查看可用指令{Style.RESET_ALL}")
    
    def print_help(self):
        """顯示幫助信息"""
        print(f"\n{Fore.CYAN}📖 VTuber AI LLM 使用說明{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}═══════════════════════════════════════{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}基本對話:{Style.RESET_ALL}")
        print("  直接輸入任何文字與AI助手對話")
        print("  系統會自動使用RAG檢索相關知識")
        
        print(f"\n{Fore.GREEN}知識庫管理:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}/add <文件路徑>{Style.RESET_ALL} - 添加文檔到知識庫")
        print(f"    支持格式: .txt, .pdf, .docx")
        print(f"    例如: /add data/documents/manual.pdf")
        
        print(f"\n{Fore.GREEN}搜索功能:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}/search <關鍵詞>{Style.RESET_ALL} - 搜索知識庫內容")
        print(f"    例如: /search 安裝教程")
        
        print(f"\n{Fore.GREEN}RAG控制:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}/rag on{Style.RESET_ALL}  - 啟用RAG檢索 (默認開啟)")
        print(f"  {Fore.CYAN}/rag off{Style.RESET_ALL} - 禁用RAG檢索")
        
        print(f"\n{Fore.GREEN}打字模擬:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}/typing on{Style.RESET_ALL}  - 啟用人性化打字模擬")
        print(f"  {Fore.CYAN}/typing off{Style.RESET_ALL} - 禁用打字模擬")
        print(f"  {Fore.CYAN}/typing status{Style.RESET_ALL} - 查看打字模擬狀態")
        
        print(f"\n{Fore.GREEN}系統信息:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}/stats{Style.RESET_ALL} - 顯示知識庫統計信息")
        print(f"  {Fore.CYAN}/model{Style.RESET_ALL} - 顯示模型詳細信息")
        print(f"  {Fore.CYAN}/clear{Style.RESET_ALL} - 清空整個知識庫")
        print()
    
    async def add_document(self, file_path: str):
        """添加文檔到知識庫"""
        try:
            if not file_path:
                print(f"{Fore.RED}❌ 請指定文件路徑{Style.RESET_ALL}")
                print(f"{Fore.CYAN}例如: /add data/documents/manual.pdf{Style.RESET_ALL}")
                return
            
            path = Path(file_path)
            if not path.exists():
                print(f"{Fore.RED}❌ 文件不存在: {file_path}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.YELLOW}📄 正在處理文檔: {path.name}...{Style.RESET_ALL}")
            
            result = await self.core_service.add_document(file_path)
            
            if result['success']:
                print(f"{Fore.GREEN}✅ {result['message']}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ {result.get('error', '添加失敗')}{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ 添加文檔時發生錯誤: {e}{Style.RESET_ALL}")
    
    async def search_knowledge_base(self, query: str):
        """搜索知識庫"""
        try:
            if not query:
                print(f"{Fore.RED}❌ 請輸入搜索關鍵詞{Style.RESET_ALL}")
                return
            
            print(f"{Fore.YELLOW}🔍 搜索中...{Style.RESET_ALL}")
            
            result = await self.core_service.search_knowledge_base(query, top_k=5)
            
            if not result['success']:
                print(f"{Fore.RED}❌ 搜索失敗: {result.get('error')}{Style.RESET_ALL}")
                return
            
            results = result['results']
            if not results:
                print(f"{Fore.YELLOW}📭 未找到相關內容{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.GREEN}🔍 搜索結果 (找到 {result['count']} 條):{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
            for i, result_item in enumerate(results, 1):
                similarity = result_item['similarity']
                source = result_item['metadata'].get('filename', '未知來源')
                content = result_item['content'][:200] + "..." if len(result_item['content']) > 200 else result_item['content']
                
                print(f"\n{Fore.YELLOW}[{i}] 來源: {source} (相關度: {similarity:.2f}){Style.RESET_ALL}")
                print(f"{content}")
                
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 搜索時發生錯誤: {e}{Style.RESET_ALL}")
    
    async def show_stats(self):
        """顯示系統統計信息"""
        try:
            stats = self.core_service.get_stats()
            
            if not stats.get('success'):
                print(f"{Fore.RED}❌ 獲取統計信息失敗: {stats.get('error')}{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.CYAN}📊 系統統計信息{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*30}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}知識庫文檔數量: {stats['total_documents']}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}集合名稱: {stats['collection_name']}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}RAG狀態: {'啟用' if stats['rag_enabled'] else '禁用'}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}活躍用戶數: {stats['active_users']}{Style.RESET_ALL}")
            
            # 檢查文檔目錄
            docs_dir = Path("data/documents")
            if docs_dir.exists():
                file_count = len([f for f in docs_dir.rglob("*") if f.is_file()])
                print(f"{Fore.GREEN}上傳文件數量: {file_count}{Style.RESET_ALL}")
            
            print(f"{Fore.YELLOW}{'='*30}{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 獲取統計信息失敗: {e}{Style.RESET_ALL}")
    
    async def show_model_info(self):
        """顯示模型詳細信息"""
        try:
            print(f"\n{Fore.CYAN}🤖 模型詳細信息{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}")
            
            # 獲取模型信息
            model_info = self.core_service.get_model_info()
            
            if "error" in model_info:
                print(f"{Fore.RED}❌ 獲取模型信息失敗: {model_info['error']}{Style.RESET_ALL}")
                return
            
            # 顯示主模型信息
            llm_info = model_info.get('llm_model', {})
            print(f"\n{Fore.GREEN}📦 主模型 (LLM):{Style.RESET_ALL}")
            print(f"  模型類型: {llm_info.get('model_type', 'Unknown')}")
            print(f"  量化方式: {llm_info.get('quantization', 'Unknown')}")
            print(f"  運行設備: {llm_info.get('device', 'Unknown')}")
            
            # 顯示嵌入模型信息
            emb_info = model_info.get('embedding_model', {})
            print(f"\n{Fore.GREEN}🔍 嵌入模型:{Style.RESET_ALL}")
            print(f"  模型類型: {emb_info.get('model_type', 'Unknown')}")
            print(f"  量化方式: {emb_info.get('quantization', 'Unknown')}")
            print(f"  運行設備: {emb_info.get('device', 'Unknown')}")
            
            # 顯示GPU信息（如果可用）
            if 'gpu_total' in llm_info:
                print(f"\n{Fore.GREEN}🎮 GPU 記憶體使用:{Style.RESET_ALL}")
                print(f"  總記憶體: {llm_info.get('gpu_total', 'Unknown')}")
                print(f"  已分配: {llm_info.get('gpu_allocated', 'Unknown')}")
                print(f"  已緩存: {llm_info.get('gpu_cached', 'Unknown')}")
            
            # 顯示量化優勢
            print(f"\n{Fore.GREEN}⚡ 量化優勢:{Style.RESET_ALL}")
            print(f"  🔹 主模型4bit: 記憶體使用減少約75%")
            print(f"  🔹 嵌入模型8bit: 記憶體使用減少約50%")
            print(f"  🔹 推理速度大幅提升")
            print(f"  🔹 保持模型精度")
            print(f"  🔹 支援更大批次處理")
            
            print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 顯示模型信息失敗: {e}{Style.RESET_ALL}")
    
    async def clear_knowledge_base(self):
        """清空知識庫"""
        try:
            print(f"{Fore.YELLOW}⚠️ 確定要清空整個知識庫嗎? (y/N): {Style.RESET_ALL}", end="")
            confirm = input().strip().lower()
            
            if confirm in ['y', 'yes', '是']:
                print(f"{Fore.YELLOW}🗑️ 正在清空知識庫...{Style.RESET_ALL}")
                result = await self.core_service.clear_knowledge_base()
                
                if result['success']:
                    print(f"{Fore.GREEN}✅ {result['message']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ {result.get('error', '清空知識庫失敗')}{Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}操作已取消{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ 清空知識庫時發生錯誤: {e}{Style.RESET_ALL}")
    
    async def show_conversation_memory(self):
        """顯示對話記憶狀態"""
        try:
            memory_status = self.core_service.get_user_memory_status(self.user_id)
            
            if not memory_status.get('success'):
                print(f"{Fore.RED}❌ 獲取記憶狀態失敗: {memory_status.get('error')}{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.CYAN}💭 對話記憶狀態{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*30}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}記憶輪數: {memory_status['memory_count']}/{memory_status['max_length']}{Style.RESET_ALL}")
            
            history = memory_status.get('history', [])
            if history:
                stats = self.core_service.get_stats()
                character_name = stats.get('character_name', 'AI助手')
                
                print(f"\n{Fore.GREEN}最近的對話:{Style.RESET_ALL}")
                for i, (user_msg, bot_response) in enumerate(history, 1):
                    user_short = user_msg[:50] + "..." if len(user_msg) > 50 else user_msg
                    bot_short = bot_response[:50] + "..." if len(bot_response) > 50 else bot_response
                    print(f"  {Fore.CYAN}[{i}] 用戶: {user_short}{Style.RESET_ALL}")
                    print(f"  {Fore.MAGENTA}[{i}] {character_name}: {bot_short}{Style.RESET_ALL}")
                    print()
            else:
                print(f"{Fore.YELLOW}暫無對話記憶{Style.RESET_ALL}")
            
            print(f"{Fore.YELLOW}{'='*30}{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 顯示對話記憶失敗: {e}{Style.RESET_ALL}")
    
    async def clear_conversation_memory(self):
        """清除對話記憶"""
        try:
            memory_status = self.core_service.get_user_memory_status(self.user_id)
            
            if memory_status['memory_count'] == 0:
                print(f"{Fore.YELLOW}💭 對話記憶已經是空的{Style.RESET_ALL}")
                return
            
            print(f"{Fore.YELLOW}⚠️ 確定要清除對話記憶嗎? (y/N): {Style.RESET_ALL}", end="")
            confirm = input().strip().lower()
            
            if confirm in ['y', 'yes', '是']:
                result = self.core_service.clear_user_memory(self.user_id)
                
                if result['success']:
                    print(f"{Fore.GREEN}✅ {result['message']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ {result.get('error', '清除記憶失敗')}{Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}操作已取消{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ 清除對話記憶時發生錯誤: {e}{Style.RESET_ALL}")
    
    async def handle_traditional_chinese_command(self, command: str):
        """處理簡繁轉換指令"""
        try:
            parts = command.split()
            if len(parts) < 2:
                await self.show_s2t_status()
                return
            
            action = parts[1].lower()
            
            if action == "on":
                result = self.core_service.toggle_traditional_chinese(True)
                if result['success']:
                    print(f"{Fore.GREEN}✅ {result['message']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}⚠️ {result['message']}{Style.RESET_ALL}")
                    
            elif action == "off":
                result = self.core_service.toggle_traditional_chinese(False)
                if result['success']:
                    print(f"{Fore.YELLOW}⚠️ {result['message']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ {result.get('error')}{Style.RESET_ALL}")
                    
            elif action == "status":
                await self.show_s2t_status()
                
            else:
                print(f"{Fore.RED}❌ 無效參數: {action}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}使用: /s2t on|off|status{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ 處理簡繁轉換指令時發生錯誤: {e}{Style.RESET_ALL}")
    
    async def show_s2t_status(self):
        """顯示簡繁轉換狀態"""
        try:
            status = self.core_service.get_traditional_chinese_status()
            
            print(f"\n{Fore.CYAN}🔄 簡繁轉換狀態{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*30}{Style.RESET_ALL}")
            
            if not status.get('success'):
                print(f"{Fore.RED}❌ {status.get('error')}{Style.RESET_ALL}")
                return
            
            # OpenCC 可用性
            opencc_status = "✅ 已安裝" if status['opencc_available'] else "❌ 未安裝"
            print(f"{Fore.GREEN}OpenCC 狀態: {opencc_status}{Style.RESET_ALL}")
            
            # 轉換器初始化
            converter_status = "✅ 已初始化" if status['converter_initialized'] else "❌ 未初始化"
            print(f"{Fore.GREEN}轉換器狀態: {converter_status}{Style.RESET_ALL}")
            
            # 功能啟用狀態
            enabled_status = "✅ 已啟用" if status['conversion_enabled'] else "❌ 已禁用"
            print(f"{Fore.GREEN}轉換功能: {enabled_status}{Style.RESET_ALL}")
            
            # 配置文件
            if status.get('config_file'):
                print(f"{Fore.GREEN}配置文件: {status['config_file']}{Style.RESET_ALL}")
            
            # 測試轉換
            test_result = status.get('test_result')
            if test_result:
                print(f"\n{Fore.CYAN}轉換測試:{Style.RESET_ALL}")
                print(f"  原文: {test_result['original']}")
                print(f"  轉換: {test_result['converted']}")
            
            print(f"{Fore.YELLOW}{'='*30}{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 顯示簡繁轉換狀態失敗: {e}{Style.RESET_ALL}")
    
    async def handle_typing_command(self, command: str):
        """處理打字模擬指令"""
        try:
            parts = command.split()
            if len(parts) < 2:
                await self.show_typing_status()
                return
            
            action = parts[1].lower()
            
            if action == "on":
                result = self.core_service.toggle_typing_simulation(True)
                print(f"{Fore.GREEN}✅ 已啟用人性化打字模擬{Style.RESET_ALL}")
                    
            elif action == "off":
                result = self.core_service.toggle_typing_simulation(False)
                print(f"{Fore.YELLOW}⚠️ 已禁用打字模擬（將直接顯示完整回應）{Style.RESET_ALL}")
                    
            elif action == "status":
                await self.show_typing_status()
                
            else:
                print(f"{Fore.RED}❌ 無效參數: {action}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}使用: /typing on|off|status{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ 處理打字模擬指令時發生錯誤: {e}{Style.RESET_ALL}")
    
    async def show_typing_status(self):
        """顯示打字模擬狀態"""
        try:
            stats = self.core_service.get_stats()
            typing_info = stats.get('typing_simulation', {})
            
            print(f"\n{Fore.CYAN}⌨️ 打字模擬狀態{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'='*30}{Style.RESET_ALL}")
            
            # 啟用狀態
            enabled = typing_info.get('enabled', False)
            status_text = "✅ 已啟用" if enabled else "❌ 已禁用"
            print(f"{Fore.GREEN}打字模擬: {status_text}{Style.RESET_ALL}")
            
            if enabled:
                # 配置參數
                speed = typing_info.get('typing_speed', 1.2)
                variation = typing_info.get('typing_speed_variation', 0.3)
                min_delay = typing_info.get('typing_min_delay', 0.5)
                max_delay = typing_info.get('typing_max_delay', 2.0)
                
                print(f"{Fore.GREEN}基礎速度: {speed:.1f}秒/行{Style.RESET_ALL}")
                print(f"{Fore.GREEN}速度變化: ±{variation:.1f}秒{Style.RESET_ALL}")
                print(f"{Fore.GREEN}延遲範圍: {min_delay:.1f}-{max_delay:.1f}秒{Style.RESET_ALL}")
                
                # 顯示預設選項
                print(f"\n{Fore.CYAN}可用預設:{Style.RESET_ALL}")
                print(f"  slow (慢速): 2.0秒/行")
                print(f"  normal (正常): 1.2秒/行")
                print(f"  fast (快速): 0.8秒/行")
                print(f"  very_fast (極快): 0.4秒/行")
                print(f"  thoughtful (深思): 1.5秒/行")
            
            print(f"{Fore.YELLOW}{'='*30}{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 顯示打字模擬狀態失敗: {e}{Style.RESET_ALL}")
    
    async def chat_loop(self):
        """主要聊天循環"""
        while self.running:
            try:
                # 獲取角色名稱
                stats = self.core_service.get_stats()
                character_name = stats.get('character_name', 'AI助手')
                
                # 顯示提示符
                print(f"{Fore.MAGENTA}{character_name}>{Style.RESET_ALL} ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # 處理指令
                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                    continue
                
                # 處理聊天消息
                await self.handle_chat(user_input)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}👋 檢測到 Ctrl+C，正在退出...{Style.RESET_ALL}")
                self.running = False
            except EOFError:
                print(f"\n{Fore.YELLOW}👋 檢測到 EOF，正在退出...{Style.RESET_ALL}")
                self.running = False
            except Exception as e:
                print(f"{Fore.RED}❌ 處理消息時發生錯誤: {e}{Style.RESET_ALL}")
                self.logger.error(f"聊天循環錯誤: {e}")
    
    async def handle_chat(self, user_input: str):
        """處理聊天 - 使用流式打字模擬"""
        print(f"{Fore.YELLOW}🤔 思考中...{Style.RESET_ALL}")
        
        try:
            # 使用新的流式生成方法
            response_parts = []
            character_name = "AI助手"
            memory_info = ""
            
            async for chunk in self.core_service.generate_response_with_typing(self.user_id, user_input):
                if chunk.get('type') == 'thinking':
                    # 更新思考狀態
                    print(f"\r{Fore.YELLOW}{chunk.get('content', '🤔 思考中...')}{Style.RESET_ALL}", end="", flush=True)
                    
                elif chunk.get('type') == 'response_start':
                    # 清除思考提示，顯示角色名
                    print(f"\r{' ' * 50}\r", end="")  # 清除之前的思考提示
                    character_name = chunk.get('character_name', 'AI助手')
                    print(f"\n{Fore.CYAN}{character_name}: {Style.RESET_ALL}", end="", flush=True)
                    
                elif chunk.get('type') == 'response_chunk':
                    # 逐步顯示回應內容
                    content = chunk.get('content', '')
                    response_parts.append(content)
                    print(content, end="", flush=True)
                    
                elif chunk.get('type') == 'response_complete':
                    # 回應完成，顯示記憶信息
                    memory_info = f"{chunk.get('conversation_length', 0)}/{chunk.get('max_length', 10)}"
                    
                    # 如果有直接的完整回應（打字模擬禁用時）
                    if 'response' in chunk:
                        print(f"\r{' ' * 50}\r", end="")  # 清除思考提示
                        character_name = chunk.get('character_name', 'AI助手')
                        response = chunk['response']
                        print(f"\n{Fore.CYAN}{character_name}: {Style.RESET_ALL}{response}")
                    
                    print(f"\n\n{Fore.YELLOW}💭 對話記憶: {memory_info} 輪{Style.RESET_ALL}")
                    break
                    
                elif chunk.get('type') == 'error':
                    # 錯誤處理
                    print(f"\r{' ' * 50}\r", end="")  # 清除思考提示
                    print(f"{Fore.RED}❌ 生成回應失敗: {chunk.get('error', '未知錯誤')}{Style.RESET_ALL}")
                    return
                    
        except Exception as e:
            print(f"\r{' ' * 50}\r", end="")  # 清除思考提示
            print(f"{Fore.RED}❌ 處理聊天時發生錯誤: {e}{Style.RESET_ALL}")
            self.logger.error(f"聊天處理錯誤: {e}")


async def main():
    """主函數"""
    try:
        # 載入配置
        config = load_config()
        
        
        # 設置日誌
        setup_logger(config['system']['log_dir'])
        
        # 檢查模型文件
        llm_path = Path(config['models']['llm']['model_path'])
        embedding_path = Path(config['models']['embedding']['model_path'])
        
        if not llm_path.exists():
            print(f"❌ LLM 模型路徑不存在: {llm_path}")
            print("請確保已下載 Qwen-8B 模型到指定路徑")
            sys.exit(1)
            
        if not embedding_path.exists():
            print(f"❌ 嵌入模型路徑不存在: {embedding_path}")
            print("請確保已下載 Qwen3-Embedding-0.6B 模型到指定路徑")
            sys.exit(1)
        
        # 初始化終端界面
        terminal = VTuberTerminal(config)
        await terminal.initialize()
        
        # 顯示歡迎信息
        terminal.print_welcome()
        
        # 開始聊天循環
        await terminal.chat_loop()
        
        # 清理資源
        if terminal.core_service:
            terminal.core_service.cleanup()
        
        print(f"{Fore.GREEN}✅ VTuber AI LLM 系統已安全關閉{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 程序被用戶中斷{Style.RESET_ALL}")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
