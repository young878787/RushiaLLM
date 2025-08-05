#!/usr/bin/env python3
"""
VTuber AI CustomTkinter GUI 啟動器
基於現有main.py的GUI版本
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import yaml

def check_dependencies():
    """檢查並安裝所需依賴"""
    try:
        import customtkinter
        print("✅ CustomTkinter 已安裝")
    except ImportError:
        print("❌ CustomTkinter 未安裝")
        print("正在安裝 CustomTkinter...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
            print("✅ CustomTkinter 安裝完成")
        except Exception as e:
            print(f"❌ 自動安裝失敗: {e}")
            print("請手動運行: pip install customtkinter")
            sys.exit(1)

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


async def main():
    """主函數"""
    try:
        print("🚀 啟動 VTuber AI CustomTkinter GUI...")
        
        # 檢查依賴
        check_dependencies()
        
        # 現在才導入GUI模組（確保依賴已安裝）
        from src.core_service import VTuberCoreService
        from src.utils.logger import setup_logger
        from gui import VTuberCustomGUI
        
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
        
        print("⏳ 初始化核心服務...")
        
        # 初始化核心服務
        core_service = VTuberCoreService(config)
        
        if await core_service.initialize():
            print("✅ 核心服務初始化完成")
            
            # 創建並運行GUI
            print("🎨 啟動 CustomTkinter GUI 界面...")
            gui = VTuberCustomGUI(core_service)
            
            # 顯示啟動完成信息
            print("✅ GUI 界面已啟動")
            print("📱 GUI 功能說明:")
            print("   - 💬 聊天室: 支持打字模擬效果的對話")
            print("   - 🧠 RAG 控制: 一鍵切換知識庫檢索")
            print("   - 📚 文檔管理: 上傳文檔到知識庫")
            print("   - 🔍 知識搜索: 測試知識庫搜索功能")
            print("   - ⚙️ 系統監控: 查看模型和系統狀態")
            print("   - 🎛️ 智能控制: 打字速度、換行等設置")
            print("\n🎯 開始使用 VTuber AI GUI 吧！")
            
            # 運行GUI主循環
            gui.run()
            
            # 清理資源
            core_service.cleanup()
            print("✅ VTuber AI GUI 已安全關閉")
        else:
            print("❌ 核心服務初始化失敗")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n👋 程序被用戶中斷")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 在Windows上設置事件循環策略
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
