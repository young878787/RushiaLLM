#!/usr/bin/env python3
"""
實時語音轉文字服務 (RealtimeSTT)
基於 GitHub 上的 RealtimeSTT 庫實現即時語音轉錄
支援多種語言和自訂配置
"""
import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import queue
import json

try:
    from RealtimeSTT import AudioToTextRecorder
    REALTIME_STT_AVAILABLE = True
except ImportError:
    REALTIME_STT_AVAILABLE = False
    print("⚠️  RealtimeSTT 未安裝，請運行: pip install RealtimeSTT")

try:
    import opencc
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False
    print("⚠️  OpenCC 未安裝，請運行: pip install opencc")

# 支援的語言
class STTLanguage(Enum):
    CHINESE_TRADITIONAL = "zh"  # Whisper 使用 zh 來處理中文（包含繁體和簡體）
    CHINESE_SIMPLIFIED = "zh"   # Whisper 使用 zh 來處理中文（包含繁體和簡體）
    CHINESE_CANTONESE = "yue"   # 粵語
    ENGLISH = "en"
    JAPANESE = "ja"
    KOREAN = "ko"

# STT 引擎類型
class STTEngine(Enum):
    WHISPER_TINY = "tiny"
    WHISPER_BASE = "base"
    WHISPER_SMALL = "small"
    WHISPER_MEDIUM = "medium"
    WHISPER_LARGE = "large"
    WHISPER_LARGE_V2 = "large-v2"
    WHISPER_LARGE_V3 = "large-v3"

@dataclass
class TranscriptionResult:
    """轉錄結果數據類"""
    text: str
    confidence: float
    language: str
    timestamp: datetime
    audio_duration: float
    is_final: bool = True
    words: List[Dict] = None

@dataclass 
class STTConfig:
    """STT 配置類"""
    # 基本配置
    language: STTLanguage = STTLanguage.CHINESE_TRADITIONAL
    model: STTEngine = STTEngine.WHISPER_BASE
    
    # 音頻配置
    sample_rate: int = 16000
    chunk_size: int = 1024
    
    # 即時處理配置
    wake_words: List[str] = None  # 喚醒詞
    wake_words_sensitivity: float = 0.6
    
    # 語音檢測配置
    silero_sensitivity: float = 0.4  # 語音活動檢測靈敏度 (RealtimeSTT 預設值)
    webrtc_sensitivity: int = 3  # WebRTC VAD 靈敏度 (0-3)
    post_speech_silence_duration: float = 0.6  # 語音後靜音時長 (RealtimeSTT 預設值)
    min_length_of_recording: float = 0.5  # 最短錄音時長 (RealtimeSTT 預設值)
    min_gap_between_recordings: float = 0.0  # 錄音間最短間隔
    
    # 即時轉錄配置  
    enable_realtime_transcription: bool = False  # 啟用即時轉錄 (預設關閉以避免複雜性)
    realtime_processing_pause: float = 0.2  # 即時處理間隔 (RealtimeSTT 預設值)
    realtime_model_type: str = "base"  # 即時轉錄模型
    
    # GPU 配置
    use_gpu: bool = True
    gpu_device_index: Optional[int] = 0  # RealtimeSTT 預設值
    
    # 文字轉換配置
    enable_opencc: bool = True  # 啟用 OpenCC 簡轉繁
    opencc_config: str = "s2twp.json"  # OpenCC 配置文件
    
    # 其他配置
    beam_size: int = 5
    initial_prompt: Optional[str] = None


class RealtimeSTTService:
    """基於 RealtimeSTT 的即時語音轉文字服務"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        
        # 檢查依賴
        if not REALTIME_STT_AVAILABLE:
            raise ImportError("RealtimeSTT 庫未安裝，請先安裝: pip install RealtimeSTT")
        
        # 載入配置
        self.config = self._load_config(config or {})
        
        # 核心組件
        self.recorder: Optional[AudioToTextRecorder] = None
        self.is_listening = False
        self.is_initialized = False
        
        # 事件回調
        self.transcription_callbacks: List[Callable] = []
        self.recording_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        self._stop_callbacks: List[Callable] = []
        
        # 統計資料
        self.stats = {
            "total_recordings": 0,
            "total_transcriptions": 0,
            "total_audio_duration": 0.0,
            "start_time": None,
            "last_transcription": None,
            "error_count": 0
        }
        
        # 即時轉錄狀態
        self.realtime_transcription_enabled = self.config.enable_realtime_transcription
        self.realtime_text_buffer = ""
        
        # OpenCC 轉換器
        self.opencc_converter = None
        if self.config.enable_opencc and OPENCC_AVAILABLE:
            try:
                self.opencc_converter = opencc.OpenCC(self.config.opencc_config)
                self.logger.info(f"✅ OpenCC 初始化成功，使用配置: {self.config.opencc_config}")
            except Exception as e:
                self.logger.error(f"OpenCC 初始化失敗: {e}")
                self.opencc_converter = None
        elif self.config.enable_opencc and not OPENCC_AVAILABLE:
            self.logger.warning("OpenCC 已啟用但未安裝，請運行: pip install opencc")
        
        # 線程安全
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.listening_thread: Optional[threading.Thread] = None
    
    def _load_config(self, user_config: Dict[str, Any]) -> STTConfig:
        """載入和驗證配置"""
        config_dict = {}
        
        # 從用戶配置中提取 STT 相關設定
        stt_config = user_config.get('stt', {})
        
        # 語言設定
        language_str = stt_config.get('language', 'zh-TW')
        try:
            # 映射用戶輸入到實際的 Whisper 語言代碼
            language_mapping = {
                'zh-TW': STTLanguage.CHINESE_TRADITIONAL,
                'zh-CN': STTLanguage.CHINESE_SIMPLIFIED,
                'zh': STTLanguage.CHINESE_TRADITIONAL,
                'yue': STTLanguage.CHINESE_CANTONESE,
                'en': STTLanguage.ENGLISH,
                'ja': STTLanguage.JAPANESE,
                'ko': STTLanguage.KOREAN
            }
            config_dict['language'] = language_mapping.get(language_str, STTLanguage.CHINESE_TRADITIONAL)
        except ValueError:
            self.logger.warning(f"不支援的語言: {language_str}，使用預設值")
            config_dict['language'] = STTLanguage.CHINESE_TRADITIONAL
        
        # 模型設定
        model_str = stt_config.get('model', 'base')
        try:
            config_dict['model'] = STTEngine(model_str)
        except ValueError:
            self.logger.warning(f"不支援的模型: {model_str}，使用預設值")
            config_dict['model'] = STTEngine.WHISPER_BASE
        
        # 音頻配置
        config_dict['sample_rate'] = stt_config.get('sample_rate', 16000)
        config_dict['chunk_size'] = stt_config.get('chunk_size', 1024)
        
        # 語音檢測配置
        config_dict['silero_sensitivity'] = stt_config.get('silero_sensitivity', 0.4)  # RealtimeSTT 預設值
        config_dict['webrtc_sensitivity'] = stt_config.get('webrtc_sensitivity', 3)
        config_dict['post_speech_silence_duration'] = stt_config.get('post_speech_silence_duration', 0.6)  # RealtimeSTT 預設值
        config_dict['min_length_of_recording'] = stt_config.get('min_length_of_recording', 0.5)  # RealtimeSTT 預設值
        
        # 即時轉錄配置
        config_dict['enable_realtime_transcription'] = stt_config.get('enable_realtime_transcription', False)  # 預設關閉
        config_dict['realtime_processing_pause'] = stt_config.get('realtime_processing_pause', 0.2)  # RealtimeSTT 預設值
        config_dict['realtime_model_type'] = stt_config.get('realtime_model_type', 'tiny')
        
        # GPU 配置
        config_dict['use_gpu'] = stt_config.get('use_gpu', True)
        config_dict['gpu_device_index'] = stt_config.get('gpu_device_index', 0)  # RealtimeSTT 預設值
        
        # OpenCC 配置
        config_dict['enable_opencc'] = stt_config.get('enable_opencc', True)  # 預設啟用
        config_dict['opencc_config'] = stt_config.get('opencc_config', 's2twp.json')  # 簡轉繁（台灣用詞）
        
        # 喚醒詞配置
        wake_words = stt_config.get('wake_words', [])
        if wake_words:
            config_dict['wake_words'] = wake_words
            config_dict['wake_words_sensitivity'] = stt_config.get('wake_words_sensitivity', 0.6)
        
        return STTConfig(**config_dict)
    
    async def initialize(self) -> bool:
        """初始化 STT 服務"""
        try:
            self.logger.info("🎤 初始化 RealtimeSTT 服務...")
            
            # 準備 RealtimeSTT 配置
            recorder_config = {
                "model": self.config.model.value,
                "language": self.config.language.value,
                "silero_sensitivity": self.config.silero_sensitivity,
                "webrtc_sensitivity": self.config.webrtc_sensitivity,
                "post_speech_silence_duration": self.config.post_speech_silence_duration,
                "min_length_of_recording": self.config.min_length_of_recording,
                "min_gap_between_recordings": self.config.min_gap_between_recordings,
                "enable_realtime_transcription": self.config.enable_realtime_transcription,
                "realtime_processing_pause": self.config.realtime_processing_pause,
                "realtime_model_type": self.config.realtime_model_type,
                "on_recording_start": self._on_recording_start,
                "on_recording_stop": self._on_recording_stop,
                "on_transcription_start": self._on_transcription_start,
                "beam_size": self.config.beam_size,
                "sample_rate": self.config.sample_rate,
                "device": "cuda" if self.config.use_gpu else "cpu"
            }
            
            # 可選配置
            if self.config.gpu_device_index is not None:
                recorder_config["gpu_device_index"] = self.config.gpu_device_index
            
            if self.config.initial_prompt:
                recorder_config["initial_prompt"] = self.config.initial_prompt
            
            if self.config.wake_words:
                recorder_config["wake_words"] = " ".join(self.config.wake_words)  # RealtimeSTT 期望字符串而非列表
                recorder_config["wake_words_sensitivity"] = self.config.wake_words_sensitivity
            
            # 創建 AudioToTextRecorder
            self.recorder = AudioToTextRecorder(**recorder_config)
            
            self.is_initialized = True
            self.stats["start_time"] = datetime.now()
            
            self.logger.info(f"✅ RealtimeSTT 初始化完成")
            self.logger.info(f"   - 模型: {self.config.model.value}")
            self.logger.info(f"   - 語言: {self.config.language.value}")
            self.logger.info(f"   - GPU: {'啟用' if self.config.use_gpu else '禁用'}")
            self.logger.info(f"   - 即時轉錄: {'啟用' if self.config.enable_realtime_transcription else '禁用'}")
            self.logger.info(f"   - OpenCC 簡轉繁: {'啟用' if self.opencc_converter else '禁用'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"RealtimeSTT 初始化失敗: {e}")
            return False
    
    def start_listening(self) -> bool:
        """開始語音監聽"""
        if not self.is_initialized:
            self.logger.error("STT 服務未初始化")
            return False

        if self.is_listening:
            self.logger.warning("STT 服務已在監聽中")
            return True

        try:
            # 確保停止事件已清除
            self._stop_event.clear()
            
            # 清理緩存，避免顯示上一次的結果
            self.realtime_text_buffer = ""
            self.logger.debug("清理 realtime_text_buffer")

            # 設置監聽狀態
            self.is_listening = True

            # 重置監聽線程（如果存在舊線程）
            if self.listening_thread and self.listening_thread.is_alive():
                self.logger.warning("發現活躍的舊監聽線程，等待其結束...")
                self.listening_thread.join(timeout=1.0)

            # 啟動新的監聽線程
            self.listening_thread = threading.Thread(target=self._listening_loop, daemon=True)
            self.listening_thread.start()

            self.logger.info("🎤 開始語音監聽...")
            return True

        except Exception as e:
            self.logger.error(f"啟動語音監聽失敗: {e}")
            self.is_listening = False
            return False

    def stop_listening(self) -> bool:
        """停止語音監聽"""
        try:
            if not self.is_listening:
                self.logger.debug("語音監聽已經停止")
                return True
            
            self.logger.info("正在停止語音監聽...")
            
            # 首先設置停止標誌
            self.is_listening = False
            self._stop_event.set()
            
            # RealtimeSTT 的正確停止順序很重要
            try:
                if self.recorder:
                    self.logger.debug("正在停止 RealtimeSTT 錄音器...")
                    
                    # 1. 首先嘗試 abort() - 立即中止當前處理
                    if hasattr(self.recorder, 'abort'):
                        self.logger.debug("調用 abort() 停止當前處理...")
                        self.recorder.abort()
                    
                    # 2. 然後調用 stop() - 正常停止錄音
                    if hasattr(self.recorder, 'stop'):
                        self.logger.debug("調用 stop() 停止錄音...")
                        self.recorder.stop()
                    
                    # 3. 不調用 shutdown() - 留給 cleanup() 處理
                    # shutdown() 可能會阻塞，所以我們跳過它
                    self.logger.debug("跳過 shutdown() 調用以避免阻塞")
                        
                    self.logger.debug("RealtimeSTT 停止序列完成")
            except Exception as e:
                self.logger.warning(f"停止 RealtimeSTT 時出現錯誤: {e}")
            
            # 等待監聽線程結束
            if self.listening_thread and self.listening_thread.is_alive():
                self.logger.debug("等待監聽線程結束...")
                
                # 縮短等待時間，避免卡住太久
                join_timeout = 0.5  # 只等待0.5秒
                self.listening_thread.join(timeout=join_timeout)
                
                if self.listening_thread.is_alive():
                    self.logger.debug(f"監聽線程未在 {join_timeout} 秒內結束，設為後台完成")
                    # 不要阻塞，讓線程在後台自然結束
                else:
                    self.logger.debug("監聽線程已正常結束")
            
            # 重置狀態
            self._stop_event.clear()
            self.listening_thread = None
            
            # 觸發停止回調通知GUI
            for callback in self._stop_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"停止回調執行失敗: {e}")
            
            self.logger.info("🔇 語音監聽已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止語音監聽失敗: {e}")
            # 即使出錯也要確保狀態正確
            self.is_listening = False
            self._stop_event.set()
            return False
    
    def _listening_loop(self):
        """主要監聽循環"""
        self.logger.info("STT 監聽循環啟動")
        
        try:
            while self.is_listening and not self._stop_event.is_set():
                try:
                    self.logger.debug("等待語音輸入...")
                    
                    # 在調用 text() 之前再次檢查停止標誌
                    if not self.is_listening or self._stop_event.is_set():
                        self.logger.debug("監聽循環收到停止信號，準備退出")
                        break
                    
                    # RealtimeSTT的text()方法是阻塞的，但會在沒有音頻時快速返回空字符串
                    try:
                        transcription = self.recorder.text()
                    except Exception as text_error:
                        # text()方法被中斷或出錯
                        if "abort" in str(text_error).lower() or "stop" in str(text_error).lower():
                            self.logger.debug(f"text() 方法被中斷: {text_error}")
                            break
                        else:
                            self.logger.warning(f"text() 方法異常: {text_error}")
                            # 短暫休息後繼續
                            time.sleep(0.1)
                            continue
                    
                    # text()返回後立即檢查是否需要停止
                    if not self.is_listening or self._stop_event.is_set():
                        self.logger.debug("監聽已停止，忽略轉錄結果")
                        break
                    
                    if transcription and transcription.strip():
                        self.logger.debug(f"收到轉錄結果: {transcription}")
                        
                        # 處理轉錄前再次檢查狀態
                        if not self.is_listening or self._stop_event.is_set():
                            self.logger.debug("處理轉錄前發現停止信號，跳過處理")
                            break
                            
                        # 處理轉錄結果
                        self._process_transcription(transcription.strip())
                    else:
                        self.logger.debug("收到空轉錄結果，繼續監聽")
                        # 短暫休息避免空循環消耗CPU
                        if self.is_listening and not self._stop_event.is_set():
                            time.sleep(0.01)
                    
                    # 檢查即時轉錄（如果啟用）
                    if self.config.enable_realtime_transcription and not self._stop_event.is_set():
                        try:
                            realtime_text = getattr(self.recorder, 'realtime_text', '')
                            if realtime_text and realtime_text != self.realtime_text_buffer:
                                if self.is_listening and not self._stop_event.is_set():
                                    self._process_realtime_text(realtime_text)
                                    self.realtime_text_buffer = realtime_text
                        except AttributeError:
                            # realtime_text 屬性可能不存在
                            pass
                        except Exception as e:
                            self.logger.debug(f"即時轉錄處理錯誤: {e}")
                    
                except Exception as e:
                    # 捕獲單次處理的錯誤，但不中斷整個循環
                    self.logger.error(f"語音處理錯誤: {e}")
                    self._trigger_error_callbacks(str(e))
                    
                    # 如果是嚴重錯誤或收到停止信號，退出循環
                    if not self.is_listening or self._stop_event.is_set():
                        break
                        
                    # 短暫休息後繼續
                    time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"監聽循環嚴重錯誤: {e}")
            self._trigger_error_callbacks(str(e))
        finally:
            self.logger.info("STT 監聽循環結束")
            # 確保在循環結束時清理狀態
            self.is_listening = False
    
    def _process_transcription(self, text: str):
        """處理完整轉錄結果"""
        try:
            with self._lock:
                self.stats["total_transcriptions"] += 1
                self.stats["last_transcription"] = datetime.now()
            
            self.logger.debug(f"開始處理轉錄: {text}")
            
            # 應用 OpenCC 轉換（簡轉繁）
            converted_text = self._convert_text_with_opencc(text)
            
            # 創建轉錄結果
            result = TranscriptionResult(
                text=converted_text,
                confidence=0.95,  # RealtimeSTT 通常不提供信心分數
                language=self.config.language.value,
                timestamp=datetime.now(),
                audio_duration=0.0,  # 需要從錄音器獲取
                is_final=True
            )
            
            self.logger.info(f"📝 轉錄完成: {converted_text}")
            if converted_text != text:
                self.logger.debug(f"   原文: {text}")
                self.logger.debug(f"   轉換後: {converted_text}")
            
            # 觸發回調
            self.logger.debug(f"觸發 {len(self.transcription_callbacks)} 個轉錄回調")
            self._trigger_transcription_callbacks(result)
            
        except Exception as e:
            self.logger.error(f"轉錄處理失敗: {e}")
            self._trigger_error_callbacks(str(e))
    
    def _process_realtime_text(self, text: str):
        """處理即時轉錄文字"""
        try:
            # 應用 OpenCC 轉換（簡轉繁）
            converted_text = self._convert_text_with_opencc(text)
            
            # 創建即時轉錄結果
            result = TranscriptionResult(
                text=converted_text,
                confidence=0.8,  # 即時轉錄信心度較低
                language=self.config.language.value,
                timestamp=datetime.now(),
                audio_duration=0.0,
                is_final=False  # 即時轉錄不是最終結果
            )
            
            self.logger.debug(f"⚡ 即時轉錄: {converted_text}")
            
            # 觸發回調（可能需要特殊處理即時結果）
            self._trigger_realtime_callbacks(result)
            
        except Exception as e:
            self.logger.error(f"處理即時轉錄失敗: {e}")
    
    def _convert_text_with_opencc(self, text: str) -> str:
        """使用 OpenCC 轉換文字（簡轉繁）"""
        if not self.opencc_converter or not text.strip():
            return text
        
        try:
            converted = self.opencc_converter.convert(text)
            return converted
        except Exception as e:
            self.logger.error(f"OpenCC 轉換失敗: {e}")
            return text  # 轉換失敗時返回原文
    
    # ==================== 事件回調 ====================
    
    def _on_recording_start(self):
        """錄音開始回調"""
        with self._lock:
            self.stats["total_recordings"] += 1
        
        self.logger.debug("🔴 開始錄音")
        self._trigger_recording_callbacks("recording_start", {"timestamp": datetime.now()})
    
    def _on_recording_stop(self):
        """錄音停止回調"""
        self.logger.debug("⚫ 停止錄音")
        self._trigger_recording_callbacks("recording_stop", {"timestamp": datetime.now()})
    
    def _on_transcription_start(self, *args):
        """轉錄開始回調"""
        self.logger.debug("📝 開始轉錄")
        self._trigger_recording_callbacks("transcription_start", {"timestamp": datetime.now()})
    
    def _trigger_transcription_callbacks(self, result: TranscriptionResult):
        """觸發轉錄回調"""
        for callback in self.transcription_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # 在新的線程中運行異步回調
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 如果事件循環正在運行，使用 call_soon_threadsafe
                            loop.call_soon_threadsafe(self._schedule_async_callback, callback, result)
                        else:
                            # 如果事件循環沒有運行，創建新任務
                            asyncio.create_task(callback(result))
                    except RuntimeError:
                        # 沒有事件循環，在新線程中運行
                        threading.Thread(
                            target=self._run_async_callback, 
                            args=(callback, result), 
                            daemon=True
                        ).start()
                else:
                    callback(result)
            except Exception as e:
                self.logger.error(f"轉錄回調執行失敗: {e}")
    
    def _schedule_async_callback(self, callback, *args):
        """在事件循環中安排異步回調"""
        asyncio.create_task(callback(*args))
    
    def _run_async_callback(self, callback, *args):
        """在新事件循環中運行異步回調"""
        try:
            asyncio.run(callback(*args))
        except Exception as e:
            self.logger.error(f"異步回調執行失敗: {e}")
    
    def _trigger_realtime_callbacks(self, result: TranscriptionResult):
        """觸發即時轉錄回調"""
        # 可以有專門的即時轉錄回調，或者復用轉錄回調
        self._trigger_transcription_callbacks(result)
    
    def _trigger_recording_callbacks(self, event_type: str, data: Dict):
        """觸發錄音事件回調"""
        for callback in self.recording_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # 在新的線程中運行異步回調
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 如果事件循環正在運行，使用 call_soon_threadsafe
                            loop.call_soon_threadsafe(self._schedule_async_recording_callback, callback, event_type, data)
                        else:
                            # 如果事件循環沒有運行，創建新任務
                            asyncio.create_task(callback(event_type, data))
                    except RuntimeError:
                        # 沒有事件循環，在新線程中運行
                        threading.Thread(
                            target=self._run_async_recording_callback, 
                            args=(callback, event_type, data), 
                            daemon=True
                        ).start()
                else:
                    callback(event_type, data)
            except Exception as e:
                self.logger.error(f"錄音回調執行失敗: {e}")
    
    def _schedule_async_recording_callback(self, callback, event_type, data):
        """在事件循環中安排異步錄音回調"""
        asyncio.create_task(callback(event_type, data))
    
    def _run_async_recording_callback(self, callback, event_type, data):
        """在新事件循環中運行異步錄音回調"""
        try:
            asyncio.run(callback(event_type, data))
        except Exception as e:
            self.logger.error(f"異步錄音回調執行失敗: {e}")
    
    def _trigger_error_callbacks(self, error_message: str):
        """觸發錯誤回調"""
        with self._lock:
            self.stats["error_count"] += 1
        
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # 在新的線程中運行異步回調
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 如果事件循環正在運行，使用 call_soon_threadsafe
                            loop.call_soon_threadsafe(self._schedule_async_error_callback, callback, error_message)
                        else:
                            # 如果事件循環沒有運行，創建新任務
                            asyncio.create_task(callback(error_message))
                    except RuntimeError:
                        # 沒有事件循環，在新線程中運行
                        threading.Thread(
                            target=self._run_async_error_callback, 
                            args=(callback, error_message), 
                            daemon=True
                        ).start()
                else:
                    callback(error_message)
            except Exception as e:
                self.logger.error(f"錯誤回調執行失敗: {e}")
    
    def _schedule_async_error_callback(self, callback, error_message):
        """在事件循環中安排異步錯誤回調"""
        asyncio.create_task(callback(error_message))
    
    def _run_async_error_callback(self, callback, error_message):
        """在新事件循環中運行異步錯誤回調"""
        try:
            asyncio.run(callback(error_message))
        except Exception as e:
            self.logger.error(f"異步錯誤回調執行失敗: {e}")
    
    # ==================== 回調註冊 ====================
    
    def add_transcription_callback(self, callback: Callable[[TranscriptionResult], None]):
        """添加轉錄結果回調"""
        self.transcription_callbacks.append(callback)
    
    def add_recording_callback(self, callback: Callable[[str, Dict], None]):
        """添加錄音事件回調"""
        self.recording_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str], None]):
        """添加錯誤回調"""
        self.error_callbacks.append(callback)
    
    def add_stop_callback(self, callback: Callable[[], None]):
        """添加停止監聽回調"""
        self._stop_callbacks.append(callback)
    
    def remove_transcription_callback(self, callback: Callable):
        """移除轉錄回調"""
        if callback in self.transcription_callbacks:
            self.transcription_callbacks.remove(callback)
    
    def remove_recording_callback(self, callback: Callable):
        """移除錄音回調"""
        if callback in self.recording_callbacks:
            self.recording_callbacks.remove(callback)
    
    def remove_error_callback(self, callback: Callable):
        """移除錯誤回調"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def remove_stop_callback(self, callback: Callable):
        """移除停止回調"""
        if callback in self._stop_callbacks:
            self._stop_callbacks.remove(callback)
    
    # ==================== 配置管理 ====================
    
    def update_sensitivity(self, silero_sensitivity: float = None, webrtc_sensitivity: int = None) -> bool:
        """更新語音檢測靈敏度"""
        try:
            if not self.recorder:
                return False
            
            if silero_sensitivity is not None:
                self.config.silero_sensitivity = silero_sensitivity
                # RealtimeSTT 可能需要重新初始化才能應用新設定
            
            if webrtc_sensitivity is not None:
                self.config.webrtc_sensitivity = webrtc_sensitivity
            
            self.logger.info(f"更新靈敏度設定: Silero={self.config.silero_sensitivity}, WebRTC={self.config.webrtc_sensitivity}")
            return True
            
        except Exception as e:
            self.logger.error(f"更新靈敏度失敗: {e}")
            return False
    
    def update_language(self, language: STTLanguage) -> bool:
        """更新語言設定（需要重新初始化）"""
        try:
            self.config.language = language
            self.logger.info(f"語言設定已更新為: {language.value}")
            return True
        except Exception as e:
            self.logger.error(f"更新語言設定失敗: {e}")
            return False
    
    def toggle_realtime_transcription(self, enabled: bool) -> bool:
        """切換即時轉錄功能"""
        try:
            self.realtime_transcription_enabled = enabled
            self.config.enable_realtime_transcription = enabled
            self.logger.info(f"即時轉錄已{'啟用' if enabled else '禁用'}")
            return True
        except Exception as e:
            self.logger.error(f"切換即時轉錄失敗: {e}")
            return False
    
    # ==================== 狀態查詢 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計資料"""
        with self._lock:
            stats = self.stats.copy()
        
        if stats["start_time"]:
            stats["uptime_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
        
        stats.update({
            "is_listening": self.is_listening,
            "is_initialized": self.is_initialized,
            "realtime_transcription_enabled": self.realtime_transcription_enabled,
            "current_language": self.config.language.value,
            "current_model": self.config.model.value,
            "callback_counts": {
                "transcription": len(self.transcription_callbacks),
                "recording": len(self.recording_callbacks),
                "error": len(self.error_callbacks)
            }
        })
        
        return stats
    
    def get_config_info(self) -> Dict[str, Any]:
        """獲取配置資訊"""
        return {
            "language": self.config.language.value,
            "model": self.config.model.value,
            "sample_rate": self.config.sample_rate,
            "silero_sensitivity": self.config.silero_sensitivity,
            "webrtc_sensitivity": self.config.webrtc_sensitivity,
            "post_speech_silence_duration": self.config.post_speech_silence_duration,
            "min_length_of_recording": self.config.min_length_of_recording,
            "enable_realtime_transcription": self.config.enable_realtime_transcription,
            "realtime_model_type": self.config.realtime_model_type,
            "use_gpu": self.config.use_gpu,
            "gpu_device_index": self.config.gpu_device_index,
            "enable_opencc": self.config.enable_opencc,
            "opencc_config": self.config.opencc_config,
            "opencc_available": self.opencc_converter is not None,
            "wake_words": self.config.wake_words,
            "wake_words_sensitivity": self.config.wake_words_sensitivity if self.config.wake_words else None
        }
    
    def is_ready(self) -> bool:
        """檢查服務是否就緒"""
        return self.is_initialized and REALTIME_STT_AVAILABLE
    
    # ==================== 清理資源 ====================
    
    def cleanup(self):
        """清理資源 - 極速版本，跳過可能阻塞的操作"""
        try:
            self.logger.info("開始清理 STT 服務資源...")
            
            # 立即重置所有狀態 - 這些操作絕對不會阻塞
            self.is_listening = False
            self.is_initialized = False
            self._stop_event.set()
            
            # 立即清理回調列表
            self.transcription_callbacks.clear()
            self.recording_callbacks.clear()
            self.error_callbacks.clear()
            self._stop_callbacks.clear()
            
            # 清理線程引用
            self.listening_thread = None
            
            # 對於 RealtimeSTT 錄音器：完全異步處理，主線程不等待
            if self.recorder:
                self.logger.debug("將 RealtimeSTT 清理完全移至後台...")
                
                # 保存引用供後台處理
                recorder_ref = self.recorder
                
                # 立即清空主引用 - 這是關鍵！
                self.recorder = None
                
                # 啟動完全獨立的後台處理
                import threading
                import weakref
                
                def ultra_async_cleanup():
                    """超級異步清理 - 在完全獨立的線程中處理"""
                    try:
                        # 嘗試設置停止標誌（快速操作）
                        quick_flags = ['_stop_requested', 'running', '_is_running']
                        for flag in quick_flags:
                            if hasattr(recorder_ref, flag):
                                try:
                                    if 'stop' in flag:
                                        setattr(recorder_ref, flag, True)
                                    else:
                                        setattr(recorder_ref, flag, False)
                                except:
                                    pass
                        
                        # 嘗試快速停止（相對快速）
                        for method in ['abort', 'stop']:
                            if hasattr(recorder_ref, method):
                                try:
                                    getattr(recorder_ref, method)()
                                except:
                                    pass
                        
                        # shutdown 可能很慢，但在後台執行不影響主線程
                        if hasattr(recorder_ref, 'shutdown'):
                            try:
                                recorder_ref.shutdown()
                            except:
                                pass
                                
                    except Exception:
                        # 靜默處理所有異常，不影響主程序
                        pass
                    finally:
                        # 清理引用
                        try:
                            del recorder_ref
                        except:
                            pass
                
                # 創建守護進程線程，不會阻塞程序退出
                cleanup_thread = threading.Thread(
                    target=ultra_async_cleanup,
                    daemon=True,  # 關鍵：守護線程
                    name="RealtimeSTT_UltraCleanup"
                )
                cleanup_thread.start()
                
                # 主線程不等待，立即繼續
                self.logger.debug("RealtimeSTT 後台清理已啟動，主線程立即返回")
            
            # 最終狀態重置
            self._stop_event.clear()
            
            # 主線程立即完成
            self.logger.info("✅ STT 服務立即清理完成")
            
        except Exception as e:
            self.logger.error(f"STT 服務清理失敗: {e}")
        finally:
            # 絕對確保這些狀態正確
            self.recorder = None
            self.is_listening = False
            self.is_initialized = False


# ==================== 便利函數 ====================

async def create_stt_service(config: Dict[str, Any] = None) -> RealtimeSTTService:
    """創建並初始化 STT 服務的便利函數"""
    service = RealtimeSTTService(config)
    success = await service.initialize()
    
    if not success:
        raise RuntimeError("STT 服務初始化失敗")
    
    return service

def test_stt_service():
    """測試 STT 服務的基本功能"""
    import asyncio
    
    async def transcription_handler(result: TranscriptionResult):
        print(f"轉錄結果: {result.text}")
        print(f"語言: {result.language}, 信心度: {result.confidence:.2f}")
        print(f"時間: {result.timestamp.strftime('%H:%M:%S')}")
        print("-" * 50)
    
    async def recording_handler(event_type: str, data: Dict):
        print(f"錄音事件: {event_type} at {data['timestamp'].strftime('%H:%M:%S')}")
    
    async def error_handler(error: str):
        print(f"錯誤: {error}")
    
    async def main():
        try:
            # 創建服務
            config = {
                'stt': {
                    'language': 'zh-TW',
                    'model': 'base',  # 使用 base 模型
                    'enable_realtime_transcription': False,  # 先關閉即時轉錄
                    'silero_sensitivity': 0.4,  # 使用預設值
                    'use_gpu': True,  # 使用 GPU 加速
                    'enable_opencc': True,  # 啟用 OpenCC 簡轉繁
                    'opencc_config': 's2twp.json'  # 簡體轉繁體（台灣用詞）
                }
            }
            
            service = await create_stt_service(config)
            
            # 註冊回調
            service.add_transcription_callback(transcription_handler)
            service.add_recording_callback(recording_handler)
            service.add_error_callback(error_handler)
            
            # 顯示配置
            print("STT 服務配置:")
            config_info = service.get_config_info()
            for key, value in config_info.items():
                print(f"  {key}: {value}")
            print("-" * 50)
            
            # 開始監聽
            print("開始語音監聽，請說話...")
            print("提示：STT 服務會持續監聽，每次說話都會即時轉錄")
            print("     測試將運行30秒後自動停止，實際使用時可以無限期運行")
            print("     每次檢測到語音都會立即處理，不受時間限制")
            service.start_listening()
            
            # 測試運行 30 秒（實際使用時可以無限期運行）
            await asyncio.sleep(30)
            
            # 停止並顯示統計
            service.stop_listening()
            
            print("\n統計資料:")
            stats = service.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # 清理
            service.cleanup()
            
        except Exception as e:
            print(f"測試失敗: {e}")
    
    # 運行測試
    asyncio.run(main())


def test_continuous_listening():
    """演示持續監聽功能（適合實際使用）"""
    import asyncio
    
    async def transcription_handler(result: TranscriptionResult):
        print(f"🗣️  轉錄: {result.text}")
        print(f"   時間: {result.timestamp.strftime('%H:%M:%S')}")
        
        # 可以在這裡加入自動響應邏輯
        if "停止" in result.text or "結束" in result.text:
            print("檢測到停止指令，準備結束...")
            return "stop"  # 返回停止信號
    
    async def recording_handler(event_type: str, data: Dict):
        if event_type == "recording_start":
            print("🔴 開始錄音...")
        elif event_type == "recording_stop":
            print("⚫ 錄音結束，處理中...")
    
    async def error_handler(error: str):
        print(f"❌ 錯誤: {error}")
    
    async def main():
        print("持續監聽演示 - STT 服務整合示例")
        print("=" * 60)
        print("說 '停止' 或 '結束' 來終止程序")
        print("=" * 60)
        
        try:
            # 創建服務配置
            config = {
                'stt': {
                    'language': 'zh-TW',
                    'model': 'base',
                    'enable_realtime_transcription': False,
                    'silero_sensitivity': 0.4,
                    'use_gpu': True,
                    'enable_opencc': True,
                    'opencc_config': 's2twp.json',
                    'post_speech_silence_duration': 0.6,  # 調整靜音檢測時間
                    'min_length_of_recording': 0.5
                }
            }
            
            service = await create_stt_service(config)
            
            # 註冊回調
            service.add_transcription_callback(transcription_handler)
            service.add_recording_callback(recording_handler)
            service.add_error_callback(error_handler)
            
            # 開始持續監聽
            service.start_listening()
            
            # 持續運行直到用戶說停止
            print("\n🎤 開始持續監聽...")
            try:
                while service.is_listening:
                    await asyncio.sleep(0.1)  # 檢查間隔
            except KeyboardInterrupt:
                print("\n收到中斷信號，正在停止...")
            
            # 停止服務
            service.stop_listening()
            
            print("\n📊 最終統計:")
            stats = service.get_stats()
            print(f"   總錄音次數: {stats['total_recordings']}")
            print(f"   總轉錄次數: {stats['total_transcriptions']}")
            print(f"   運行時間: {stats.get('uptime_seconds', 0):.1f} 秒")
            print(f"   錯誤次數: {stats['error_count']}")
            
            # 清理資源
            service.cleanup()
            
        except Exception as e:
            print(f"❌ 演示失敗: {e}")
            import traceback
            traceback.print_exc()
    
    # 運行演示
    asyncio.run(main())


if __name__ == "__main__":
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("RealtimeSTT 服務測試")
    print("=" * 50)
    print("1. test_stt_service() - 30秒限時測試")
    print("2. test_continuous_listening() - 持續監聽演示")
    print("=" * 50)
    
    if REALTIME_STT_AVAILABLE:
        # 可以選擇運行哪個測試
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "continuous":
            test_continuous_listening()
        else:
            test_stt_service()
    else:
        print("請先安裝 RealtimeSTT: pip install RealtimeSTT")
