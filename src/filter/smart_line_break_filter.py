#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智慧換行處理器
根據語義轉折、問候、情感表達等邏輯自動為回應添加換行，讓回覆更像真人
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from .base_filter import BaseResponseFilter

# 使用統一的日誌配置
try:
    from ..log_config import setup_module_logging
    logger = setup_module_logging(__name__, startup_mode=True)
except ImportError:
    logger = logging.getLogger(__name__)

class SmartLineBreakFilter(BaseResponseFilter):
    """智慧換行處理器 - 讓回應更自然地分段"""
    
    def __init__(self, chat_instance=None):
        super().__init__(chat_instance)
        
        # 問候語模式
        self.greeting_patterns = [
            r'(你好|哈囉|嗨|hi|hello|早安|午安|晚安|おはよう|こんにちは|こんばんは)',
            r'(歡迎|回來|辛苦了|回家了)',
            r'(今天|現在|這時候|剛才)'
        ]
        
        # 情感表達模式
        self.emotion_patterns = [
            r'(真的|確實|當然|沒錯|是啊|對呢)',
            r'(好棒|太好了|真棒|厲害|讚|不錯)',
            r'(好可愛|好甜|好溫柔|好暖)',
            r'(抱歉|對不起|不好意思|sorry)',
            r'(謝謝|感謝|thank)',
            r'(哈哈|呵呵|嘻嘻|aha|哈|呵)'
        ]
        
        # 語義轉折詞
        self.transition_patterns = [
            r'(不過|但是|可是|然而|只是|雖然|儘管)',
            r'(所以|因此|那麼|這樣|於是|結果)',
            r'(另外|還有|而且|再說|順便|對了)',
            r'(其實|實際上|說實話|老實說|坦白說)',
            r'(總之|總而言之|簡單說|反正|anyway)'
        ]
        
        # 問句模式
        self.question_patterns = [
            r'([？?])',
            r'(要不要|要嗎|對嗎|是嗎|好嗎|如何|怎麼樣)',
            r'(什麼|哪個|哪裡|為什麼|怎麼|when|where|what|why|how)'
        ]
        
        # 建議或邀請模式
        self.suggestion_patterns = [
            r'(建議|推薦|可以|不如|要不|或許)',
            r'(一起|我們|來吧|走吧|去吧)',
            r'(試試|看看|聽聽|想想|考慮)'
        ]
        
        # 親密表達模式
        self.intimate_patterns = [
            r'(親愛的|寶貝|darling|honey)',
            r'(愛你|喜歡你|miss you|想你)',
            r'(抱抱|親親|摸摸|蹭蹭|♪|♡|💕|❤️)'
        ]
        
        # 時間相關模式
        self.time_patterns = [
            r'(現在|剛才|等等|之後|稍後|later)',
            r'(今天|明天|昨天|yesterday|today|tomorrow)',
            r'(早上|中午|下午|晚上|深夜|morning|evening|night)'
        ]
        
        # 不應該在前面換行的詞
        self.no_break_before = [
            r'(了|呢|啊|哦|喔|哩|嘛|呀|耶|哇)',
            r'(的|地|得)',
            r'(♪|♡|～|~)',
            r'([。！？，；、.!?,:;])'
        ]
        
        # 編譯正則表達式模式
        self._compile_patterns()
        
        logger.debug("SmartLineBreakFilter 智慧換行處理器初始化完成")
    
    def _compile_patterns(self):
        """編譯正則表達式模式以提高效能"""
        self.compiled_patterns = {
            'greeting': [re.compile(pattern, re.IGNORECASE) for pattern in self.greeting_patterns],
            'emotion': [re.compile(pattern, re.IGNORECASE) for pattern in self.emotion_patterns],
            'transition': [re.compile(pattern, re.IGNORECASE) for pattern in self.transition_patterns],
            'question': [re.compile(pattern, re.IGNORECASE) for pattern in self.question_patterns],
            'suggestion': [re.compile(pattern, re.IGNORECASE) for pattern in self.suggestion_patterns],
            'intimate': [re.compile(pattern, re.IGNORECASE) for pattern in self.intimate_patterns],
            'time': [re.compile(pattern, re.IGNORECASE) for pattern in self.time_patterns],
            'no_break_before': [re.compile(pattern, re.IGNORECASE) for pattern in self.no_break_before]
        }
    
    def _should_filter(self, response: str, user_input: str = "", context: Dict = None) -> bool:
        """檢查是否需要進行換行處理"""
        if not response or len(response.strip()) < 10:  # 太短的回應不需要換行
            return False
        
        # 如果已經有適當的換行，且不是過長的單行，可能不需要處理
        lines = response.split('\n')
        if len(lines) > 1:
            # 檢查是否有過長的行
            has_long_lines = any(len(line.strip()) > 50 for line in lines)
            if not has_long_lines:
                return False
        
        # 檢查是否有多個句子（即使較短也應該處理）
        sentence_count = len([s for s in response.split('？') + response.split('！') + response.split('。') + response.split('♪') + response.split('♡') if s.strip()])
        if sentence_count > 1:
            return True
        
        # 檢查是否包含問候語、問句、情感表達等需要換行的元素
        has_special_elements = (
            any(pattern.search(response) for pattern in self.compiled_patterns['greeting']) or
            any(pattern.search(response) for pattern in self.compiled_patterns['question']) or
            any(pattern.search(response) for pattern in self.compiled_patterns['emotion']) or
            any(pattern.search(response) for pattern in self.compiled_patterns['intimate'])
        )
        
        return has_special_elements or len(response.strip()) >= 20
    
    def _apply_filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """應用智慧換行處理"""
        if not response:
            return response
        
        logger.debug(f"開始智慧換行處理: {response[:30]}...")
        
        # 先移除現有的換行，重新處理
        cleaned_response = ' '.join(response.split())
        
        # 找出所有可能的換行點
        break_points = self._find_break_points(cleaned_response)
        
        # 根據換行點重新組織文本
        formatted_response = self._apply_line_breaks(cleaned_response, break_points)
        
        logger.debug(f"換行處理完成: {len(break_points)} 個換行點")
        return formatted_response
    
    def smart_sentence_split(self, response: str) -> List[str]:
        """
        智慧分句處理 - 將回應分割成自然的句子
        使用更安全的邏輯避免標點錯位
        """
        if not response:
            return []
        
        # 移除多餘的空白和換行
        response = ' '.join(response.split()).strip()
        
        # 使用更安全的分句方法
        sentences = []
        current = ""
        
        i = 0
        while i < len(response):
            char = response[i]
            current += char
            
            # 檢查是否為分句點
            should_split = False
            next_start = i + 1
            
            # 1. 強制分句標點：。！？!?
            if char in ['。', '！', '？', '!', '?']:
                # 收集後續的表情符號和語氣詞
                while next_start < len(response) and response[next_start] in ['♪', '♡', '～', '💕', '❤️', ' ', '呢', '啊', '哦', '呀', '吧']:
                    if response[next_start] != ' ':
                        current += response[next_start]
                    next_start += 1
                should_split = True
            
            # 2. 省略號處理
            elif char == '.' and i + 2 < len(response) and response[i+1:i+3] == '..':
                current += '..'
                next_start = i + 3
                if len(current.strip()) >= 6:
                    remaining = response[next_start:].strip()
                    if len(remaining) > 5:
                        should_split = True
            
            # 3. 表情符號結尾（有條件分句）
            elif char in ['♪', '♡', '～', '💕', '❤️'] and len(current.strip()) >= 8:
                remaining = response[i+1:].strip()
                if len(remaining) > 10:
                    # 檢查是否後面有轉折詞
                    if any(remaining.startswith(word) for word in ['嗯', '不過', '但是', '可是', '另外', '還有', '而且']):
                        should_split = True
            
            # 4. 逗號處理（只在有明確轉折時分句）
            elif char == '，' and len(current.strip()) >= 15:
                remaining = response[i+1:].strip()
                if len(remaining) > 8:
                    # 只在有明確轉折詞時才分句
                    if any(remaining.startswith(word) for word in ['不過', '但是', '可是', '然而', '另外', '還有']):
                        should_split = True
            
            # 5. 單獨的英文句號
            elif char == '.' and len(current.strip()) >= 8:
                remaining = response[i+1:].strip()
                if len(remaining) > 5 and not remaining.startswith('.'):
                    should_split = True
            
            # 執行分句
            if should_split:
                cleaned = current.strip()
                if len(cleaned) >= 3:
                    sentences.append(cleaned)
                current = ""
                i = next_start - 1
            
            i += 1
        
        # 處理剩餘內容
        if current.strip():
            sentences.append(current.strip())
        
        # 後處理：修正標點和合併
        return self._final_process_sentences(sentences)
    
    def _find_best_cut_position(self, text: str) -> int:
        """尋找最佳的切分位置"""
        # 切分優先級：標點 > 語義詞 > 助詞 > 空格
        cut_priorities = [
            (['，', '、', ';', ','], 70, 90),  # 標點符號，在70-90%位置
            (['的', '了', '過', '著', '呢', '啊', '哦'], 60, 85),  # 助詞
            (['一起', '可以', '應該', '可能', '或者', '要不'], 50, 80),  # 語義詞
            ([' '], 50, 85),  # 空格
        ]
        
        text_len = len(text)
        best_pos = -1
        best_score = 0
        
        for markers, min_pct, max_pct in cut_priorities:
            min_pos = int(text_len * min_pct / 100)
            max_pos = int(text_len * max_pct / 100)
            
            for marker in markers:
                # 從理想位置向兩邊搜索
                ideal_pos = int(text_len * 0.75)  # 理想位置75%
                
                # 向前搜索
                for pos in range(ideal_pos, min_pos - 1, -1):
                    if pos + len(marker) <= text_len and text[pos:pos + len(marker)] == marker:
                        score = 100 - abs(pos - ideal_pos)  # 越接近理想位置分數越高
                        if score > best_score:
                            best_score = score
                            best_pos = pos + len(marker)
                
                # 向後搜索
                for pos in range(ideal_pos + 1, min(max_pos, text_len - len(marker) + 1)):
                    if text[pos:pos + len(marker)] == marker:
                        score = 100 - abs(pos - ideal_pos)
                        if score > best_score:
                            best_score = score
                            best_pos = pos + len(marker)
        
        return best_pos if best_pos > 0 else text_len // 2
    
    def _final_process_sentences(self, sentences: List[str]) -> List[str]:
        """
        最終處理句子 - 修正標點錯誤並智慧合併
        """
        if not sentences:
            return sentences
        
        # 修正每個句子的標點問題
        corrected = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 專門修正標點錯誤
            sentence = self._fix_punctuation_errors(sentence)
            
            # 檢查句子是否合理
            if len(sentence) >= 2:
                corrected.append(sentence)
        
        # 智慧合併相關句子
        final = []
        i = 0
        
        while i < len(corrected):
            current = corrected[i]
            
            # 檢查是否需要與下一句合併
            if i + 1 < len(corrected):
                next_sent = corrected[i + 1]
                
                # 合併條件
                merge = False
                total_len = len(current + next_sent)
                
                # 1. 兩句都較短且總長度合理
                if len(current) < 12 and len(next_sent) < 15 and total_len < 40:
                    merge = True
                
                # 2. 當前句以不完整詞結尾
                elif any(current.rstrip('♪♡～！？。').endswith(w) for w in ['是', '會', '要', '想', '可以', '應該', '一起', '能']):
                    if total_len < 50:
                        merge = True
                
                # 3. 語義連貫
                elif self._check_semantic_connection(current, next_sent) and total_len < 55:
                    merge = True
                
                if merge:
                    merged = current.rstrip() + ' ' + next_sent.lstrip()
                    final.append(merged)
                    i += 2
                    continue
            
            final.append(current)
            i += 1
        
        return final
    
    def _check_semantic_connection(self, sent1: str, sent2: str) -> bool:
        """檢查兩句是否有語義連接"""
        # 檢查共同主題詞
        theme_words = ['你', '我', '露醬', '今天', '一起', '去', '做', '玩', '好', '想']
        
        sent1_themes = [w for w in theme_words if w in sent1]
        sent2_themes = [w for w in theme_words if w in sent2]
        
        # 有共同主題且句子都不太長
        return len(set(sent1_themes) & set(sent2_themes)) > 0 and len(sent1) < 20 and len(sent2) < 20
    
    def _fix_punctuation_errors(self, sentence: str) -> str:
        """專門修正標點符號錯誤"""
        # 移除明顯的錯誤標點組合
        sentence = re.sub(r'啦\s*，\s*？', '啦～', sentence)
        sentence = re.sub(r'呢\s*，\s*？', '呢？', sentence) 
        sentence = re.sub(r'哦\s*，\s*？', '哦～', sentence)
        sentence = re.sub(r'啊\s*，\s*？', '啊～', sentence)
        
        # 修正孤立的問號
        sentence = re.sub(r'，\s*？\s*$', '～', sentence)  # 句尾的 "，？" -> "～"
        sentence = re.sub(r'，\s*？', '？', sentence)  # 其他的 "，？" -> "？"
        
        # 修正其他錯誤組合
        sentence = re.sub(r'，\s*！', '！', sentence)
        sentence = re.sub(r'。\s*？', '？', sentence)
        sentence = re.sub(r'。\s*！', '！', sentence)
        
        return sentence
    
    def _fix_sentence_issues(self, sentence: str) -> str:
        """修正句子中的常見問題"""
        # 修正重複的標點符號
        sentence = re.sub(r'([♪♡～])\1+', r'\1', sentence)
        
        # 修正空格問題
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # 修正中英文之間的空格
        sentence = re.sub(r'([a-zA-Z])\s+([^\sa-zA-Z])', r'\1\2', sentence)
        sentence = re.sub(r'([^\sa-zA-Z])\s+([a-zA-Z])', r'\1\2', sentence)
        
        # 修正錯位的標點符號（重點修正）
        sentence = re.sub(r'啦\s*，\s*？', '啦～', sentence)  # 修正 "啦，？" -> "啦～"
        sentence = re.sub(r'呢\s*，\s*？', '呢？', sentence)  # 修正 "呢，？" -> "呢？"
        sentence = re.sub(r'哦\s*，\s*？', '哦～', sentence)  # 修正 "哦，？" -> "哦～"
        sentence = re.sub(r'啊\s*，\s*？', '啊～', sentence)  # 修正 "啊，？" -> "啊～"
        sentence = re.sub(r'，\s*？\s*$', '～', sentence)  # 句尾的 "，？" -> "～"
        sentence = re.sub(r'，\s*？', '？', sentence)  # 其他的 "，？" -> "？"
        sentence = re.sub(r'，\s*！', '！', sentence)  # 修正 "，！" -> "！"
        sentence = re.sub(r'。\s*？', '？', sentence)  # 修正 "。？" -> "？"
        
        # 修正句子中的連續標點
        sentence = re.sub(r'[。，]\s*([？！])', r'\1', sentence)
        
        # 修正連續點號
        sentence = re.sub(r'\.{2,}', '...', sentence)  # 規範化省略號
        sentence = re.sub(r'\.{1}([^.])', r'.\1', sentence)  # 確保單點後有內容
        
        # 特殊修正：移除不應該存在的孤立問號
        sentence = re.sub(r'^[，。]\s*？', '？', sentence)  # 句首的錯位標點
        sentence = re.sub(r'，？$', '？', sentence)  # 句尾的錯位標點
        
        # 確保句子有適當的結尾
        if sentence and not re.search(r'[。！？♪♡～.!?]$', sentence):
            # 根據內容添加適當的結尾
            if any(word in sentence for word in ['嗎', '呢', '吧', '?','~']):
                if not sentence.endswith('?'):
                    sentence += '？'
            elif any(word in sentence for word in ['！', '!', '太', '真', '好', '棒']):
                if not sentence.endswith('!') and '！' not in sentence:
                    sentence += '！'
            else:
                sentence += '♪'
        
        return sentence
    
    def _find_break_points(self, text: str) -> List[Tuple[int, str, float]]:
        """
        找出所有可能的換行點
        
        Returns:
            List[Tuple[int, str, float]]: (位置, 類型, 權重)
        """
        break_points = []
        
        # 1. 問候語後換行
        for pattern in self.compiled_patterns['greeting']:
            for match in pattern.finditer(text):
                # 確保問候語後面有內容
                after_pos = match.end()
                if after_pos < len(text) - 3:
                    break_points.append((after_pos, 'greeting', 0.9))
        
        # 2. 情感表達後換行
        for pattern in self.compiled_patterns['emotion']:
            for match in pattern.finditer(text):
                after_pos = match.end()
                # 檢查後面是否有足夠內容
                if after_pos < len(text) - 5:
                    # 如果情感表達後有逗號或句號，在其後換行
                    next_char_pos = self._find_next_punctuation(text, after_pos)
                    if next_char_pos and next_char_pos < len(text) - 3:
                        break_points.append((next_char_pos, 'emotion', 0.8))
                    else:
                        break_points.append((after_pos, 'emotion', 0.7))
        
        # 3. 語義轉折前換行
        for pattern in self.compiled_patterns['transition']:
            for match in pattern.finditer(text):
                before_pos = match.start()
                # 確保轉折詞前面有內容
                if before_pos > 3:
                    break_points.append((before_pos, 'transition', 0.85))
        
        # 4. 問句後換行
        for pattern in self.compiled_patterns['question']:
            for match in pattern.finditer(text):
                after_pos = match.end()
                if after_pos < len(text) - 3:
                    break_points.append((after_pos, 'question', 0.8))
        
        # 5. 建議或邀請前換行
        for pattern in self.compiled_patterns['suggestion']:
            for match in pattern.finditer(text):
                before_pos = match.start()
                if before_pos > 5:
                    break_points.append((before_pos, 'suggestion', 0.75))
        
        # 6. 親密表達獨立成行
        for pattern in self.compiled_patterns['intimate']:
            for match in pattern.finditer(text):
                before_pos = match.start()
                after_pos = match.end()
                
                # 親密表達前換行
                if before_pos > 3:
                    break_points.append((before_pos, 'intimate_before', 0.9))
                
                # 親密表達後換行（如果後面還有內容）
                if after_pos < len(text) - 3:
                    break_points.append((after_pos, 'intimate_after', 0.9))
        
        # 7. 時間表達後換行
        for pattern in self.compiled_patterns['time']:
            for match in pattern.finditer(text):
                after_pos = match.end()
                # 找到時間表達後的適當位置
                next_punct_pos = self._find_next_punctuation(text, after_pos)
                if next_punct_pos and next_punct_pos < len(text) - 3:
                    break_points.append((next_punct_pos, 'time', 0.7))
        
        # 8. 長句子的中間點（基於句子長度）
        if len(text) > 60:
            break_points.extend(self._find_length_based_breaks(text))
        
        # 過濾掉不合適的換行點
        break_points = self._filter_break_points(text, break_points)
        
        # 按位置排序並去重
        break_points = sorted(list(set(break_points)), key=lambda x: x[0])
        
        return break_points
    
    def _find_next_punctuation(self, text: str, start_pos: int) -> Optional[int]:
        """找到下一個標點符號的位置"""
        # 支援中英文標點符號
        punct_pattern = re.compile(r'[。！？，；、.!?,:;]')
        match = punct_pattern.search(text, start_pos)
        return match.end() if match else None
    
    def _find_length_based_breaks(self, text: str) -> List[Tuple[int, str, float]]:
        """基於長度找到適當的斷行點"""
        breaks = []
        words = text.split()
        current_length = 0
        current_pos = 0
        
        for word in words:
            word_length = len(word)
            
            # 如果加上這個詞會超過理想長度，考慮在此處斷行
            if current_length + word_length > 40 and current_length > 20:
                breaks.append((current_pos, 'length', 0.5))
                current_length = word_length
            else:
                current_length += word_length + 1  # +1 for space
            
            current_pos += word_length + 1
        
        return breaks
    
    def _filter_break_points(self, text: str, break_points: List[Tuple[int, str, float]]) -> List[Tuple[int, str, float]]:
        """過濾掉不合適的換行點"""
        filtered_points = []
        
        for pos, break_type, weight in break_points:
            # 檢查是否在不應該換行的詞前面
            should_skip = False
            
            for pattern in self.compiled_patterns['no_break_before']:
                # 檢查換行點後面是否緊接著不應該換行的詞
                if pos < len(text):
                    substring = text[pos:pos+3]
                    if pattern.search(substring):
                        should_skip = True
                        break
            
            if not should_skip:
                # 確保換行點前後都有足夠的內容
                if pos > 3 and pos < len(text) - 3:
                    filtered_points.append((pos, break_type, weight))
        
        return filtered_points
    
    def _apply_line_breaks(self, text: str, break_points: List[Tuple[int, str, float]]) -> str:
        """根據換行點應用換行"""
        if not break_points:
            return text
        
        # 根據權重和距離選擇最佳換行點
        selected_breaks = self._select_optimal_breaks(text, break_points)
        
        # 應用換行
        result_parts = []
        last_pos = 0
        
        for pos, _, _ in selected_breaks:
            if pos > last_pos:
                part = text[last_pos:pos].strip()
                if part:
                    result_parts.append(part)
                last_pos = pos
        
        # 添加最後一部分
        if last_pos < len(text):
            final_part = text[last_pos:].strip()
            if final_part:
                result_parts.append(final_part)
        
        return '\n'.join(result_parts)
    
    def _select_optimal_breaks(self, text: str, break_points: List[Tuple[int, str, float]]) -> List[Tuple[int, str, float]]:
        """選擇最佳的換行點組合"""
        if not break_points:
            return []
        
        # 根據權重排序
        sorted_breaks = sorted(break_points, key=lambda x: x[2], reverse=True)
        
        selected = []
        min_distance = 15  # 兩個換行點之間的最小距離
        
        for break_point in sorted_breaks:
            pos, break_type, weight = break_point
            
            # 檢查是否與已選擇的換行點太近
            too_close = any(abs(pos - selected_pos) < min_distance 
                          for selected_pos, _, _ in selected)
            
            if not too_close:
                selected.append(break_point)
        
        # 按位置重新排序
        return sorted(selected, key=lambda x: x[0])
    
    def filter(self, response: str, user_input: str = "", context: Dict = None) -> str:
        """
        過濾器主要方法 - 實作 BaseResponseFilter 的抽象方法
        提供智慧換行和分句功能
        
        Args:
            response: 原始回應
            user_input: 用戶輸入
            context: 對話上下文
            
        Returns:
            str: 處理後的回應
        """
        try:
            self.stats['processed_count'] += 1
            
            # 檢查是否需要處理
            if not self._should_filter(response, user_input, context):
                return response
            
            # 優先使用智慧分句，然後重新組合為多行文本
            sentences = self.smart_sentence_split(response)
            
            if len(sentences) > 1:
                # 多句子的情況，根據內容決定換行策略
                filtered_response = self._optimize_multi_sentence_layout(sentences)
            else:
                # 單句子的情況，使用原有的換行邏輯
                filtered_response = self._apply_filter(response, user_input, context)
            
            # 檢查是否有修改
            if filtered_response != response:
                self.stats['modified_count'] += 1
                logger.debug(f"智慧換行處理完成: 原始 {len(response)} 字 -> 處理後 {len(filtered_response)} 字")
            
            return filtered_response
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"智慧換行處理發生錯誤: {str(e)}")
            # 發生錯誤時返回原始回應
            return response
    
    def _optimize_multi_sentence_layout(self, sentences: List[str]) -> str:
        """
        優化多句子的布局，決定哪些句子應該在同一行
        """
        if not sentences:
            return ""
        
        if len(sentences) == 1:
            return sentences[0]
        
        # 對於2-3句的情況，優先考慮分行以提高可讀性
        if len(sentences) <= 3:
            # 檢查是否有問句 - 問句應該獨立成行
            questions = [s for s in sentences if self._is_question(s)]
            if questions:
                # 有問句，全部分行
                return '\n'.join(sentences)
            
            # 檢查句子長度
            total_length = sum(len(s) for s in sentences)
            if total_length > 40:  # 總長度超過40字符，分行顯示
                return '\n'.join(sentences)
        
        # 分組策略：
        groups = []
        current_group = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # 檢查是否應該獨立成行
            should_separate = (
                self._is_emotional_expression(sentence) or
                self._is_question(sentence) or
                self._is_greeting(sentence) or
                sentence_len > 40
            )
            
            # 檢查是否可以合併到當前組
            can_merge = (
                not should_separate and
                current_length + sentence_len + 1 <= 50 and
                len(current_group) < 2 and
                len(sentences) > 3  # 只有句子較多時才合併
            )
            
            if can_merge and current_group:
                current_group.append(sentence)
                current_length += sentence_len + 1
            else:
                # 開始新組
                if current_group:
                    groups.append(' '.join(current_group))
                current_group = [sentence]
                current_length = sentence_len
        
        # 添加最後一組
        if current_group:
            groups.append(' '.join(current_group))
        
        return '\n'.join(groups)
    
    def _is_emotional_expression(self, sentence: str) -> bool:
        """判斷是否為情感表達"""
        emotional_indicators = [
            '♡', '♪', '～', '💕', '❤️',
            '好開心', '好快樂', '好愛', '好溫暖', '好甜',
            '超級', '真的', '好棒', '太好了', '厲害'
        ]
        return any(indicator in sentence for indicator in emotional_indicators)
    
    def _is_question(self, sentence: str) -> bool:
        """判斷是否為問句"""
        return ('？' in sentence or '?' in sentence or 
                any(word in sentence for word in ['嗎', '呢', '吧', '要不要', '好嗎', '對嗎']))
    
    def _is_greeting(self, sentence: str) -> bool:
        """判斷是否為問候語"""
        greetings = [
            '早安', '午安', '晚安', '你好', '哈囉', '嗨',
            '歡迎', '回來', '辛苦了', 'hello', 'hi'
        ]
        return any(greeting in sentence.lower() for greeting in greetings)
    
    def get_filter_name(self) -> str:
        """回傳過濾器名稱"""
        return "SmartLineBreak"
    
    def get_filter_description(self) -> str:
        """回傳過濾器描述"""
        return "智慧換行處理：根據語義轉折、問候、情感表達等邏輯自動換行"
    
    def _get_debug_info(self, original: str, filtered: str) -> Dict:
        """取得調試資訊"""
        original_lines = len(original.split('\n'))
        filtered_lines = len(filtered.split('\n'))
        
        return {
            'original_lines': original_lines,
            'new_lines': filtered_lines,
            'lines_added': filtered_lines - original_lines,
            'avg_line_length': sum(len(line) for line in filtered.split('\n')) / filtered_lines if filtered_lines > 0 else 0
        }
