"""
回應後處理過濾器模組
負責清理和優化 LLM 生成的回應內容
"""

import re
import logging
import os
from typing import List, Optional
from collections import Counter
from pathlib import Path
from datetime import datetime

# OpenCC 簡繁轉換
try:
    import opencc
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False
    logging.warning("OpenCC 未安裝，簡繁轉換功能將被禁用。請運行: pip install opencc-python-reimplemented")


class ResponseFilter:
    """回應過濾器類"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 過濾設置
        self.filter_config = config.get('vtuber', {}).get('response', {})
        self.max_sentence_repeat = self.filter_config.get('max_sentence_repeat', 1)
        self.filter_repetition = self.filter_config.get('filter_repetition', True)
        self.filter_incomplete = self.filter_config.get('filter_incomplete', True)
        
        # 簡繁轉換設置
        self.enable_traditional_chinese = self.filter_config.get('enable_traditional_chinese', True)
        self.opencc_config = self.filter_config.get('opencc_config', 's2twp.json')
        self.opencc_converter = None
        self._setup_opencc_converter()
        
        # VTuber 角色名稱
        self.character_name = '露西婭'  # 默認角色名稱，將由 LLM 管理器動態設置
        
        # 設置調試日誌
        self._setup_debug_logging(config)
        
        # 停止詞彙和特殊標記
        self.stop_patterns = [
            f"{self.character_name}：",
            f"{self.character_name}:",
            "用戶：",
            "用戶:",
            "Human:",
            "Assistant:",
            "參考資訊：",
            "參考資訊:",
        ]
        
        # 需要完全移除的特殊標記
        self.remove_tokens = [
            "<|endoftext|>",
            "<|system|>",
            "<|user|>", 
            "<|assistant|>",
            "<|context|>",
            "<|end_context|>",
            # Qwen3 專用標記
            "<|startofcontext|>",
            "<|session|>",
            "<|modeloutput|>",
            "<|m|>",
            "<|thinking|>",
            "<|startofmidnight|>",
            "<|beginofresponse|>",
            "<|endpoint|>",
            "<|startofthinking|>",
            "<|end|",
        ]
        
        # 輸出邊界標記（只保留此標記內的內容）
        self.boundary_token = "<|end|>"
        
        # 句子結束標記
        self.sentence_endings = ['。', '！', '？', '.', '!', '?', '~', '♪', '♡']
        
        # 免責聲明模式
        self.disclaimer_patterns = [
            # 完整的免責聲明
            r"（注：以上為角色情景演繹，並無實際意義）",
            r"（注：此為角色扮演情景劇情發展，非現實事件）",
            r"\(注：這裡展現了露西婭作為死靈魔法使的日常，同時也透露出她喜歡與他人相處的性格特點\）",
            
            # 變體模式
            r"\（注：.*?演繹.*?實際意義.*?\）",
            r"\（注：.*?扮演.*?非現實.*?\）",
            r"\(注：.*?演繹.*?實際意義.*?\)",
            r"\(注：.*?扮演.*?非現實.*?\)",
            
            # 其他可能的免責聲明
            r"注：.*?為虛構.*?內容",
            r"聲明：.*?角色扮演.*?",
            r"提醒：.*?僅為.*?娛樂",
            r"【.*?免責.*?聲明.*?】",
            r"\*.*?角色扮演.*?情節.*?\*",
            
            # 清理末尾的注釋
            r"\n\s*注：.*?$",
            r"\n\s*（注：.*?）\s*$",
            r"\n\s*\(注：.*?\)\s*$",
            
            # 額外的角色扮演提示
            r"以上.*?為角色扮演.*?",
            r"此.*?為虛構.*?對話",
            r"僅為.*?娛樂.*?目的",
            r"非真實.*?事件",
        ]
    
    def _setup_debug_logging(self, config: dict):
        """設置調試日誌"""
        try:
            # 🔥 統一：獲取日誌目錄（使用配置中的統一路徑）
            log_dir = Path(config.get('system', {}).get('log_dir', './scriptV2/LLM/logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 調試日誌文件路徑
            self.debug_log_path = log_dir / 'response_raw.log'
            
            # 每次啟動時清除舊日誌
            if self.debug_log_path.exists():
                self.debug_log_path.unlink()
            
            self.logger.info(f"調試日誌已設置: {self.debug_log_path}")
            
        except Exception as e:
            self.logger.error(f"設置調試日誌失敗: {e}")
            self.debug_log_path = None
    
    def _setup_opencc_converter(self):
        """設置 OpenCC 簡繁轉換器"""
        if not OPENCC_AVAILABLE or not self.enable_traditional_chinese:
            self.logger.info("OpenCC 簡繁轉換功能已禁用")
            return
        
        try:
            # 初始化簡體轉繁體轉換器，使用指定的配置文件
            self.opencc_converter = opencc.OpenCC(self.opencc_config)
            self.logger.info(f"✅ OpenCC 簡繁轉換器初始化成功，配置: {self.opencc_config}")
            
            # 測試轉換功能
            test_text = "你好，这是一个测试"
            converted = self.opencc_converter.convert(test_text)
            self.logger.info(f"🔄 轉換測試: '{test_text}' → '{converted}'")
            
        except Exception as e:
            self.logger.error(f"❌ OpenCC 初始化失敗: {e}")
            self.opencc_converter = None
            self.enable_traditional_chinese = False
    
    def _write_debug_log(self, stage: str, content: str):
        """寫入調試日誌"""
        if not self.debug_log_path:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"\n[{timestamp}] {stage}:\n{content}\n{'-'*50}\n"
            
            with open(self.debug_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            self.logger.error(f"寫入調試日誌失敗: {e}")
    
    def _remove_disclaimers(self, text: str) -> str:
        """移除模型自動添加的免責聲明"""
        if not text:
            return text
            
        original_text = text
        cleaned_text = text
        
        # 逐一移除匹配的免責聲明模式
        removed_disclaimers = []
        for pattern in self.disclaimer_patterns:
            matches = re.findall(pattern, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
            if matches:
                removed_disclaimers.extend(matches)
                cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 記錄移除的免責聲明
        if removed_disclaimers:
            self._write_debug_log("移除免責聲明 (REMOVE DISCLAIMERS)", 
                                f"移除的內容: {removed_disclaimers}\n原文: {original_text}\n清理後: {cleaned_text}")
            self.logger.debug(f"🚫 移除免責聲明: {len(removed_disclaimers)} 個")
        
        # 清理多餘的空行和空白
        cleaned_text = re.sub(r'\n\s*\n+', '\n', cleaned_text)
        cleaned_text = re.sub(r'^\s+|\s+$', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def convert_to_traditional_chinese(self, text: str) -> str:
        """將簡體中文轉換為繁體中文"""
        if not self.opencc_converter or not self.enable_traditional_chinese:
            return text
        
        if not text or not text.strip():
            return text
        
        try:
            # 執行簡繁轉換
            converted_text = self.opencc_converter.convert(text)
            
            # 記錄轉換結果（僅在有變化時）
            if converted_text != text:
                self._write_debug_log("簡繁轉換 (OPENCC CONVERSION)", 
                                    f"原文: {text}\n轉換後: {converted_text}\n配置: {self.opencc_config}")
                self.logger.debug(f"🔄 簡繁轉換 ({self.opencc_config}): '{text[:30]}...' → '{converted_text[:30]}...'")
            
            return converted_text
            
        except Exception as e:
            self.logger.error(f"簡繁轉換失敗: {e}")
            return text
    
    def toggle_traditional_chinese(self, enable: bool = None) -> bool:
        """切換簡繁轉換功能"""
        if enable is None:
            self.enable_traditional_chinese = not self.enable_traditional_chinese
        else:
            self.enable_traditional_chinese = enable
        
        status = "啟用" if self.enable_traditional_chinese else "禁用"
        self.logger.info(f"🔄 OpenCC 簡繁轉換功能已{status}")
        
        return self.enable_traditional_chinese
        
    def set_character_name(self, character_name: str):
        """設置角色名稱"""
        self.character_name = character_name
        # 更新過濾模式
        self.filter_patterns = [
            f"{self.character_name}：",
            f"{self.character_name}:",
            f"{self.character_name} :",
            f"{self.character_name}說：",
            f"{self.character_name}說:",
        ]
    
    def filter_response(self, response: str) -> str:
        """主要過濾方法"""
        if not response or not response.strip():
            return ""
        
        try:
            # 記錄原始回應
            self._write_debug_log("原始回應", response)
            
            # 截取第一個標記之前的內容
            filtered = self._extract_before_tokens(response)
            
            # 記錄被過濾掉的內容
            filtered_out = response.replace(filtered, "").strip() if filtered else response
            if filtered_out:
                self._write_debug_log("過濾掉的文字", filtered_out)
            
            # 最終清理
            if filtered:
                filtered = self._final_cleanup(filtered)
            
            # 🔥 新增：移除免責聲明
            if filtered:
                filtered = self._remove_disclaimers(filtered)
            
            # 🔥 新增：簡繁轉換處理
            if filtered and self.enable_traditional_chinese:
                filtered = self.convert_to_traditional_chinese(filtered)
            
            # 記錄最終輸出
            self._write_debug_log("最終輸出", filtered)
            
            return filtered.strip() if filtered else ""
            
        except Exception as e:
            self.logger.error(f"過濾回應時發生錯誤: {e}")
            self._write_debug_log("錯誤", f"過濾失敗: {e}")
            return response.strip()
    
    def _extract_before_tokens(self, text: str) -> str:
        """提取第一個特殊標記之前的內容"""
        # 新增：定義截斷標記模式
        truncate_patterns = [
            r'<\|[^|]*\|>',     # 原本的 <|...| > 格式
            r'<<\|[^|]*',       # 新增：<<|?? 格式
            r'<</[^>]*>',       # 新增：<</start> 格式
            r'<<[^>]*>',        # 新增：其他 << 開頭的標記
            r'<\|[^|]*\|',
            r'<|end|>',
            r'註釋',
            r'（注：.*?）',      # 新增：括號內的注釋
            r'注釋',
        ]
        
        # 尋找第一個匹配的截斷標記
        earliest_pos = len(text)
        matched_pattern = None
        
        for pattern in truncate_patterns:
            match = re.search(pattern, text)
            if match and match.start() < earliest_pos:
                earliest_pos = match.start()
                matched_pattern = pattern
        
        # 如果找到任何截斷標記，在該位置截斷
        if earliest_pos < len(text):
            content = text[:earliest_pos]
            self._write_debug_log("截斷標記檢測", f"使用模式 '{matched_pattern}' 在位置 {earliest_pos} 截斷")
            return content.strip()
        
        # 如果沒有找到截斷標記，檢查是否有其他停止標記
        for token in self.remove_tokens:
            if token in text:
                content = text.split(token)[0]
                self._write_debug_log("停止標記截斷", f"遇到停止標記 '{token}' 進行截斷")
                return content.strip()
        
        # 如果沒有找到任何標記，返回原始文本
        return text.strip()
    
    def _basic_cleanup(self, text: str) -> str:
        """基本清理"""
        original_text = text
        
        # 1. 處理邊界標記 - 在 <|end|> 處截斷
        if self.boundary_token in text:
            text = text.split(self.boundary_token)[0]
            self._write_debug_log("邊界標記截斷 (BOUNDARY CUT)", f"找到 {self.boundary_token}，截斷後: {text}")
        
        # 2. 移除所有特殊標記
        for token in self.remove_tokens:
            if token in text:
                text = text.replace(token, '')
                self._write_debug_log(f"移除標記 (REMOVE TOKEN)", f"移除 '{token}' 後: {text}")
        
        # 3. 通用過濾：移除所有 <|...| > 格式的標記
        matches = re.findall(r'<\|[^|]*\|>', text)
        if matches:
            self._write_debug_log("發現通用標記 (FOUND GENERIC TOKENS)", f"發現標記: {matches}")
            text = re.sub(r'<\|[^|]*\|>', '', text)
            self._write_debug_log("通用標記過濾後 (GENERIC FILTER)", text)
        
        # 4. 移除模型思考過程（通常在括號或特殊格式中）
        text = self._remove_thinking_process(text)
        
        # 5. 移除多餘的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 6. 移除停止詞彙
        for pattern in self.stop_patterns:
            if pattern in text:
                text = text.replace(pattern, '')
                self._write_debug_log(f"移除停止詞 (REMOVE STOP WORD)", f"移除 '{pattern}' 後: {text}")
        
        # 7. 移除開頭的標點符號
        text = re.sub(r'^[：:]+', '', text)
        
        return text.strip()
    
    def _remove_thinking_process(self, text: str) -> str:
        """移除模型的思考過程"""
        # 移除 <|thinking|> 標記內的所有內容
        text = re.sub(r'<\|thinking\|>.*?(?=<\||\Z)', '', text, flags=re.DOTALL)
        
        # 移除括號內的思考內容（如果很長的話）
        text = re.sub(r'\([^)]{20,}\)', '', text)
        text = re.sub(r'（[^）]{20,}）', '', text)
        
        # 移除常見的思考標記
        thinking_patterns = [
            r'讓我想想[^。]*。',
            r'我覺得[^。]*。',
            r'根據[^。]*，',
            r'基於[^。]*，',
            r'考慮到[^。]*，',
        ]
        
        for pattern in thinking_patterns:
            text = re.sub(pattern, '', text)
        
        return text
    
    def _remove_repetitions(self, text: str) -> str:
        """移除重複內容"""
        # 分割成句子
        sentences = self._split_sentences(text)
        
        if not sentences:
            return text
        
        # 移除連續重複的句子
        filtered_sentences = []
        prev_sentence = ""
        repeat_count = 0
        
        for sentence in sentences:
            sentence_clean = re.sub(r'[^\w\s]', '', sentence).strip()
            prev_clean = re.sub(r'[^\w\s]', '', prev_sentence).strip()
            
            if sentence_clean == prev_clean and sentence_clean:
                repeat_count += 1
                if repeat_count <= self.max_sentence_repeat:
                    filtered_sentences.append(sentence)
            else:
                repeat_count = 0
                filtered_sentences.append(sentence)
                prev_sentence = sentence
        
        # 移除 N-gram 重複
        result = ''.join(filtered_sentences)
        result = self._remove_ngram_repetitions(result)
        
        return result
    
    def _remove_ngram_repetitions(self, text: str, n: int = 3) -> str:
        """移除 N-gram 重複"""
        words = text.split()
        if len(words) < n * 2:
            return text
        
        # 檢測重複的 n-gram
        filtered_words = []
        i = 0
        
        while i < len(words):
            if i + n * 2 <= len(words):
                # 檢查當前 n-gram 是否與下一個 n-gram 重複
                current_ngram = words[i:i+n]
                next_ngram = words[i+n:i+n*2]
                
                if current_ngram == next_ngram:
                    # 跳過重複的 n-gram
                    filtered_words.extend(current_ngram)
                    i += n * 2
                else:
                    filtered_words.append(words[i])
                    i += 1
            else:
                filtered_words.append(words[i])
                i += 1
        
        return ' '.join(filtered_words)
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 使用正則表達式分割句子
        pattern = r'([。！？.!?~♪♡]+)'
        parts = re.split(pattern, text)
        
        sentences = []
        current_sentence = ""
        
        for part in parts:
            current_sentence += part
            if any(ending in part for ending in self.sentence_endings):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 添加剩餘部分
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def _remove_incomplete_sentences(self, text: str) -> str:
        """移除不完整的句子"""
        sentences = self._split_sentences(text)
        
        if not sentences:
            return text
        
        # 保留完整的句子
        complete_sentences = []
        
        for sentence in sentences:
            # 檢查句子是否以標點符號結尾
            if any(sentence.rstrip().endswith(ending) for ending in self.sentence_endings):
                complete_sentences.append(sentence)
            elif len(sentence.strip()) > 20:  # 如果句子較長，可能是完整的
                # 添加適當的結尾標點
                sentence = sentence.rstrip()
                if not any(sentence.endswith(ending) for ending in self.sentence_endings):
                    sentence += '。'
                complete_sentences.append(sentence)
        
        # 如果沒有完整句子，返回原文
        if not complete_sentences:
            return text
        
        return ''.join(complete_sentences)
    
    def _final_cleanup(self, text: str) -> str:
        """最終清理"""
        # 移除多餘的標點符號
        text = re.sub(r'([。！？.!?]){2,}', r'\1', text)
        
        # 移除多餘的空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除開頭和結尾的空白
        text = text.strip()
        
        # 確保以適當的標點符號結尾
        if text and not any(text.endswith(ending) for ending in self.sentence_endings):
            # 如果是問句，添加問號
            if any(word in text for word in ['嗎', '呢', '吧', '什麼', '怎麼', '為什麼', '哪裡']):
                text += '？'
            else:
                text += '。'
        
        return text
    
    def validate_response(self, response: str) -> bool:
        """驗證回應是否有效"""
        if not response or not response.strip():
            return False
        
        # 檢查最小長度
        if len(response.strip()) < 2:
            return False
        
        # 檢查是否包含有意義的內容
        meaningful_chars = re.sub(r'[^\w]', '', response)
        if len(meaningful_chars) < 2:
            return False
        
        return True
    
    def get_filter_stats(self, original: str, filtered: str) -> dict:
        """獲取過濾統計信息"""
        # 檢查是否進行了簡繁轉換
        conversion_applied = False
        if self.enable_traditional_chinese and self.opencc_converter:
            # 簡單檢測：如果包含簡體字符，可能進行了轉換
            simplified_chars = set('这个你们来说话')
            traditional_chars = set('這個你們來說話')
            
            has_traditional = any(char in filtered for char in traditional_chars)
            has_simplified = any(char in original for char in simplified_chars)
            conversion_applied = has_simplified and has_traditional
        
        # 檢查是否移除了免責聲明
        disclaimers_removed = False
        for pattern in self.disclaimer_patterns:
            if re.search(pattern, original, flags=re.IGNORECASE):
                disclaimers_removed = True
                break
        
        return {
            'original_length': len(original),
            'filtered_length': len(filtered),
            'reduction_ratio': 1 - (len(filtered) / len(original)) if original else 0,
            'original_sentences': len(self._split_sentences(original)),
            'filtered_sentences': len(self._split_sentences(filtered)),
            'traditional_chinese_enabled': self.enable_traditional_chinese,
            'conversion_applied': conversion_applied,
            'disclaimers_removed': disclaimers_removed,
            'opencc_available': OPENCC_AVAILABLE and self.opencc_converter is not None,
            'opencc_config': self.opencc_config if self.opencc_converter else None
        }
    
    def add_disclaimer_pattern(self, pattern: str):
        """添加自定義免責聲明模式"""
        if pattern not in self.disclaimer_patterns:
            self.disclaimer_patterns.append(pattern)
            self.logger.info(f"✅ 添加免責聲明模式: {pattern}")
    
    def remove_disclaimer_pattern(self, pattern: str):
        """移除免責聲明模式"""
        if pattern in self.disclaimer_patterns:
            self.disclaimer_patterns.remove(pattern)
            self.logger.info(f"🗑️ 移除免責聲明模式: {pattern}")
    
    def get_disclaimer_patterns(self) -> list:
        """獲取當前的免責聲明模式"""
        return self.disclaimer_patterns.copy()

    def get_conversion_status(self) -> dict:
        """獲取簡繁轉換狀態"""
        return {
            'opencc_available': OPENCC_AVAILABLE,
            'converter_initialized': self.opencc_converter is not None,
            'conversion_enabled': self.enable_traditional_chinese,
            'opencc_config': self.opencc_config if self.opencc_converter else None,
            'supported_configs': ['s2t.json', 's2tw.json', 's2twp.json', 's2hk.json'] if OPENCC_AVAILABLE else []
        }