#!/usr/bin/env python3
"""
安裝 RealtimeSTT 及相關依賴的腳本
"""
import subprocess
import sys
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def install_package(package_name, upgrade=False):
    """安裝 Python 包"""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package_name)
        
        logger.info(f"正在安裝 {package_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {package_name} 安裝成功")
            return True
        else:
            logger.error(f"❌ {package_name} 安裝失敗:")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"安裝 {package_name} 時發生錯誤: {e}")
        return False

def check_package(package_name):
    """檢查包是否已安裝"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """主安裝流程"""
    logger.info("🚀 開始安裝 RealtimeSTT 相關依賴...")
    
    # RealtimeSTT 相關包
    packages = [
        "RealtimeSTT",
        "torch",  # PyTorch (Whisper 需要)
        "torchaudio",  # PyTorch Audio
        "openai-whisper",  # Whisper
        "faster-whisper",  # 更快的 Whisper 實現
        "sounddevice",  # 音頻設備管理
        "numpy",  # 數值計算
        "scipy",  # 科學計算
        "librosa",  # 音頻處理
        "webrtcvad",  # WebRTC 語音活動檢測
        "silero-vad",  # Silero 語音活動檢測
        "pyaudio",  # 音頻 I/O (可能需要手動安裝)
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if package == "pyaudio":
            # PyAudio 在 Windows 上可能需要特殊處理
            logger.info("正在處理 PyAudio (Windows 可能需要額外步驟)...")
            if install_package(package):
                success_count += 1
            else:
                logger.warning("PyAudio 安裝失敗，你可能需要手動安裝:")
                logger.warning("  方法1: pip install pipwin && pipwin install pyaudio")
                logger.warning("  方法2: 下載 .whl 文件從 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        else:
            if install_package(package):
                success_count += 1
    
    logger.info(f"\n📊 安裝完成統計:")
    logger.info(f"   成功: {success_count}/{total_count}")
    logger.info(f"   失敗: {total_count - success_count}/{total_count}")
    
    # 檢查關鍵包
    logger.info("\n🔍 檢查關鍵包安裝狀態:")
    critical_packages = ["RealtimeSTT", "torch", "whisper", "sounddevice"]
    
    all_critical_installed = True
    for package in critical_packages:
        installed = check_package(package.replace("-", "_"))  # 處理包名差異
        status = "✅ 已安裝" if installed else "❌ 未安裝"
        logger.info(f"   {package}: {status}")
        if not installed:
            all_critical_installed = False
    
    if all_critical_installed:
        logger.info("\n🎉 所有關鍵包安裝成功！RealtimeSTT 應該可以正常工作。")
        logger.info("\n📝 測試 STT 服務:")
        logger.info("   cd scrpitsV2/LLM/src")
        logger.info("   python STT.py")
    else:
        logger.warning("\n⚠️  部分關鍵包未安裝，請手動解決依賴問題。")
    
    # 提供額外配置建議
    logger.info("\n💡 額外配置建議:")
    logger.info("   1. 確保麥克風權限已開啟")
    logger.info("   2. 如果使用 GPU，確保 CUDA 已正確安裝")
    logger.info("   3. 第一次運行會下載 Whisper 模型，需要網路連接")
    
    return all_critical_installed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
