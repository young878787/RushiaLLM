#!/usr/bin/env python3
"""
STT 整合示例
展示如何將 RealtimeSTT 整合到 VTuber AI 系統中
"""
import asyncio
import logging
import yaml
from pathlib import Path
from datetime import datetime

# 導入 STT 服務
from STT import RealtimeSTTService, TranscriptionResult

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class STTIntegrationDemo:
    """STT 整合演示類"""
    
    def __init__(self, config_path: str = None):
        # 載入配置
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 預設配置
            self.config = {
                'stt': {
                    'enabled': True,
                    'language': 'zh-TW',
                    'model': 'base',
                    'enable_realtime_transcription': True,
                    'silero_sensitivity': 0.05,
                    'auto_response': True
                }
            }
        
        self.stt_service = None
        self.transcription_history = []
        
        # 模擬 AI 回應系統
        self.ai_responses = [
            "我聽到了你的話，很有趣呢！",
            "嗯嗯，我明白你的意思。",
            "這個話題很棒，繼續說吧！",
            "哇，真的嗎？聽起來很厲害！",
            "我正在仔細聽你說話~"
        ]
        self.response_index = 0
    
    async def initialize(self):
        """初始化 STT 整合"""
        try:
            logger.info("🚀 初始化 STT 整合演示...")
            
            # 創建 STT 服務
            self.stt_service = RealtimeSTTService(self.config)
            success = await self.stt_service.initialize()
            
            if not success:
                logger.error("STT 服務初始化失敗")
                return False
            
            # 註冊回調函數
            self.stt_service.add_transcription_callback(self.on_transcription)
            self.stt_service.add_recording_callback(self.on_recording_event)
            self.stt_service.add_error_callback(self.on_error)
            
            logger.info("✅ STT 整合初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化失敗: {e}")
            return False
    
    async def on_transcription(self, result: TranscriptionResult):
        """處理轉錄結果的回調"""
        try:
            # 記錄轉錄歷史
            self.transcription_history.append({
                'text': result.text,
                'timestamp': result.timestamp,
                'is_final': result.is_final,
                'language': result.language,
                'confidence': result.confidence
            })
            
            # 顯示轉錄結果
            status = "✅ 完成" if result.is_final else "⚡ 即時"
            logger.info(f"🎤 {status} 轉錄: {result.text}")
            logger.info(f"   語言: {result.language}, 信心度: {result.confidence:.2f}")
            
            # 如果是最終結果且啟用自動回應
            if result.is_final and self.config.get('stt', {}).get('auto_response', False):
                await self.generate_ai_response(result.text)
                
        except Exception as e:
            logger.error(f"處理轉錄結果失敗: {e}")
    
    async def on_recording_event(self, event_type: str, data: dict):
        """處理錄音事件的回調"""
        event_messages = {
            'recording_start': '🔴 開始錄音',
            'recording_stop': '⚫ 停止錄音', 
            'transcription_start': '📝 開始轉錄'
        }
        
        message = event_messages.get(event_type, f'📻 {event_type}')
        timestamp = data.get('timestamp', datetime.now()).strftime('%H:%M:%S')
        logger.debug(f"{message} [{timestamp}]")
    
    async def on_error(self, error_message: str):
        """處理錯誤的回調"""
        logger.error(f"❌ STT 錯誤: {error_message}")
    
    async def generate_ai_response(self, user_input: str):
        """生成 AI 回應 (模擬)"""
        try:
            # 模擬 AI 思考時間
            await asyncio.sleep(0.5)
            
            # 簡單的回應邏輯
            response = self.ai_responses[self.response_index]
            self.response_index = (self.response_index + 1) % len(self.ai_responses)
            
            # 根據用戶輸入調整回應
            if "你好" in user_input or "嗨" in user_input:
                response = "你好！很高興見到你~"
            elif "再見" in user_input or "掰掰" in user_input:
                response = "再見！下次再聊~"
            elif "謝謝" in user_input:
                response = "不客氣！很高興能幫到你。"
            elif "?" in user_input or "？" in user_input:
                response = "這是個好問題，讓我想想..."
            
            logger.info(f"🤖 AI 回應: {response}")
            
            # 這裡可以整合 TTS (文字轉語音) 系統
            # await self.speak_response(response)
            
        except Exception as e:
            logger.error(f"生成 AI 回應失敗: {e}")
    
    def start_listening(self):
        """開始語音監聽"""
        if not self.stt_service:
            logger.error("STT 服務未初始化")
            return False
        
        success = self.stt_service.start_listening()
        if success:
            logger.info("🎤 語音監聽已啟動，請開始說話...")
            self.show_usage_tips()
        return success
    
    def stop_listening(self):
        """停止語音監聽"""
        if self.stt_service:
            self.stt_service.stop_listening()
            logger.info("🔇 語音監聽已停止")
    
    def show_usage_tips(self):
        """顯示使用提示"""
        logger.info("💡 使用提示:")
        logger.info("   - 正常說話，系統會自動檢測語音")
        logger.info("   - 說話結束後會自動轉錄")
        logger.info("   - 如果啟用即時轉錄，會看到即時結果")
        logger.info("   - 說 'Ctrl+C' 可以停止程序")
    
    def show_stats(self):
        """顯示統計資料"""
        if not self.stt_service:
            return
        
        stats = self.stt_service.get_stats()
        config_info = self.stt_service.get_config_info()
        
        logger.info("\n📊 STT 統計資料:")
        logger.info(f"   運行時間: {stats.get('uptime_seconds', 0):.1f} 秒")
        logger.info(f"   總錄音次數: {stats.get('total_recordings', 0)}")
        logger.info(f"   總轉錄次數: {stats.get('total_transcriptions', 0)}")
        logger.info(f"   錯誤次數: {stats.get('error_count', 0)}")
        logger.info(f"   轉錄歷史: {len(self.transcription_history)} 條")
        
        logger.info("\n⚙️  配置資訊:")
        logger.info(f"   語言: {config_info.get('language')}")
        logger.info(f"   模型: {config_info.get('model')}")
        logger.info(f"   GPU: {'啟用' if config_info.get('use_gpu') else '禁用'}")
        logger.info(f"   即時轉錄: {'啟用' if config_info.get('enable_realtime_transcription') else '禁用'}")
    
    def cleanup(self):
        """清理資源"""
        if self.stt_service:
            self.stt_service.cleanup()
            logger.info("✅ 資源清理完成")


async def main():
    """主程序"""
    logger.info("🎤 RealtimeSTT 整合演示")
    logger.info("=" * 50)
    
    # 創建演示實例
    config_path = Path(__file__).parent / "stt_config.yaml"
    demo = STTIntegrationDemo(str(config_path) if config_path.exists() else None)
    
    try:
        # 初始化
        success = await demo.initialize()
        if not success:
            logger.error("初始化失敗，程序結束")
            return
        
        # 開始監聽
        demo.start_listening()
        
        # 運行演示 (30 秒)
        logger.info(f"演示將運行 30 秒...")
        await asyncio.sleep(30)
        
        # 停止監聽
        demo.stop_listening()
        
        # 顯示統計
        demo.show_stats()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  用戶中斷程序")
    except Exception as e:
        logger.error(f"程序執行錯誤: {e}")
    finally:
        # 清理資源
        demo.cleanup()
        logger.info("👋 演示結束")


if __name__ == "__main__":
    # 檢查依賴
    try:
        from RealtimeSTT import AudioToTextRecorder
        print("✅ RealtimeSTT 已安裝")
    except ImportError:
        print("❌ RealtimeSTT 未安裝")
        print("請先運行: python install_stt_requirements.py")
        exit(1)
    
    # 運行演示
    asyncio.run(main())
