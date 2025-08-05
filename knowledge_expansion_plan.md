# Rushia 知識庫多樣性擴展計劃

## 🎯 目標
解決聊天主題單一化問題，增加對話的豐富性和多樣性，同時保持 Rushia 的角色一致性。

## 📚 多層知識庫架構設計

### Layer 1: 角色核心層 (現有)
```
rushia_wiki/
├── core.json                    # 角色核心
├── rushia_knowledge/            # Rushia 專用知識
```
**權重**: 1.4 (最高優先級，確保角色一致性)

### Layer 2: 通用知識層 (新增)
```
general_knowledge/
├── topics/                      # 通用話題庫
│   ├── daily_life/             # 日常生活
│   │   ├── food_culture.md     # 飲食文化
│   │   ├── weather_seasons.md  # 天氣季節
│   │   ├── hobbies_leisure.md  # 愛好休閒
│   │   └── health_wellness.md  # 健康生活
│   ├── entertainment/          # 娛樂文化
│   │   ├── anime_manga.md      # 動漫文化
│   │   ├── music_trends.md     # 音樂潮流
│   │   ├── movies_dramas.md    # 影視作品
│   │   └── gaming_industry.md  # 遊戲產業
│   ├── technology/             # 科技知識
│   │   ├── social_media.md     # 社交媒體
│   │   ├── digital_trends.md   # 數位趨勢
│   │   └── ai_future.md        # AI與未來
│   ├── culture_society/        # 文化社會
│   │   ├── japanese_culture.md # 日本文化
│   │   ├── festivals_events.md # 節慶活動
│   │   └── social_topics.md    # 社會話題
│   └── learning_growth/        # 學習成長
│       ├── language_study.md   # 語言學習
│       ├── skill_development.md # 技能發展
│       └── personal_growth.md  # 個人成長
```
**權重**: 1.0 (標準權重，提供話題多樣性)

### Layer 3: 情境對話層 (新增)
```
conversation_templates/
├── topic_transitions/          # 話題轉換
│   ├── natural_bridges.md     # 自然過渡語句
│   ├── curiosity_starters.md  # 好奇心引導
│   └── shared_interests.md    # 共同興趣發現
├── interactive_scenarios/     # 互動情境
│   ├── storytelling.md        # 說故事互動
│   ├── question_games.md      # 問答遊戲
│   ├── creative_challenges.md # 創意挑戰
│   └── emotional_support.md   # 情感支持
└── personality_expressions/   # 個性表達
    ├── rushia_opinions.md     # Rushia 觀點模板
    ├── reaction_patterns.md   # 反應模式
    └── curiosity_topics.md    # 好奇心話題
```
**權重**: 1.2 (中高權重，引導對話方向)

### Layer 4: 動態內容層 (可選)
```
dynamic_content/
├── seasonal_topics/           # 季節性話題
├── trending_discussions/      # 當前熱門話題
└── user_preferences/          # 用戶偏好記錄
```
**權重**: 0.8 (輔助權重，時效性內容)

## 🔧 RAG 系統優化策略

### 1. 多樣性檢索算法
```python
class DiversityEnhancedRAG:
    def __init__(self):
        self.diversity_config = {
            'max_same_category': 2,      # 同類別最多2個結果
            'min_different_layers': 2,   # 至少2個不同層級
            'topic_expansion_rate': 0.3, # 30%的結果用於話題拓展
            'curiosity_boost': 1.2,      # 好奇心話題加成
        }
    
    async def diversified_search(self, query, emotional_analysis):
        # 1. 基礎角色檢索 (確保一致性)
        character_results = await self.search_layer("rushia_core", query, weight=1.4)
        
        # 2. 話題拓展檢索 (增加多樣性)
        expansion_results = await self.topic_expansion_search(query, emotional_analysis)
        
        # 3. 情境對話檢索 (引導互動)
        scenario_results = await self.search_layer("conversation_templates", query, weight=1.2)
        
        # 4. 智能融合與排序
        return self.intelligent_merge(character_results, expansion_results, scenario_results)
```

### 2. 話題拓展機制
```python
class TopicExpansionEngine:
    def __init__(self):
        self.expansion_patterns = {
            'food': ['cooking', 'restaurants', 'cultural_food', 'health_eating'],
            'game': ['game_industry', 'esports', 'game_development', 'gaming_culture'],
            'anime': ['voice_acting', 'animation', 'manga', 'japanese_culture'],
            'music': ['instruments', 'genres', 'concerts', 'music_creation']
        }
    
    def expand_query(self, original_query, user_interests=None):
        # 基於原始查詢識別核心話題
        core_topic = self.identify_core_topic(original_query)
        
        # 生成相關話題擴展
        expanded_topics = self.expansion_patterns.get(core_topic, [])
        
        # 構建多樣化查詢
        return self.build_diversified_query(original_query, expanded_topics)
```

## 🎨 動態提示詞系統

### 1. 多樣性引導提示詞
```python
class DynamicPromptGenerator:
    def generate_diversity_prompt(self, conversation_context, available_topics):
        base_prompt = self.get_rushia_core_prompt()
        
        # 添加話題多樣性指導
        diversity_guidance = f"""
【今日話題靈感】
- 可以聊聊: {', '.join(available_topics[:3])}
- 如果對方沒有特定話題，露西亞可以主動分享或詢問
- 保持好奇心，對用戶的興趣表現出真誠的關注

【對話拓展技巧】
- 從用戶的話中找到可以深入的點
- 分享相關的個人感受或想法
- 適時提出開放性問題
- 連結到共同可能感興趣的話題
"""
        
        return base_prompt + diversity_guidance
```

### 2. 好奇心驅動機制
```python
class CuriosityEngine:
    def __init__(self):
        self.curiosity_triggers = {
            'new_topic': "露西亞對這個話題很好奇，想要了解更多",
            'user_interest': "露西亞注意到用戶的興趣，想要深入交流",
            'shared_experience': "露西亞想要分享相關的經驗或感受",
            'learning_opportunity': "露西亞把這當作學習新知識的機會"
        }
    
    def generate_curiosity_response(self, detected_topic, user_context):
        # 基於話題生成好奇心驅動的回應指導
        return self.curiosity_triggers.get(detected_topic, self.curiosity_triggers['new_topic'])
```

## 📊 實施步驟

### Phase 1: 知識庫擴展 (Week 1-2)
1. **創建通用知識架構**
   - 建立 `general_knowledge/` 目錄結構
   - 填充基礎話題內容 (每個分類 3-5 個檔案)
   
2. **設計對話模板**
   - 創建 `conversation_templates/` 
   - 編寫話題轉換和互動情境模板

### Phase 2: RAG 系統優化 (Week 2-3)  
1. **實施多樣性檢索**
   - 修改 `rag_system.py` 添加多層次檢索
   - 實現話題拓展算法
   
2. **權重系統調整**
   - 重新平衡不同知識層的權重
   - 添加多樣性評分機制

### Phase 3: 提示詞系統升級 (Week 3-4)
1. **動態提示詞生成**
   - 修改 `core.py` 的 `generate_dynamic_system_prompt`
   - 添加話題感知和好奇心機制
   
2. **對話流程優化**
   - 實現智能話題建議
   - 添加對話豐富度監控

### Phase 4: 測試與調優 (Week 4-5)
1. **多樣性測試**
   - 測試不同話題的對話效果
   - 調整權重和檢索策略
   
2. **角色一致性驗證**
   - 確保 Rushia 個性不會被稀釋
   - 微調角色表達方式

## 🎯 預期效果

### 短期目標 (1個月內)
- 對話話題覆蓋面增加 300%
- 用戶對話持續時間提升 50%
- 重複性回應減少 70%

### 長期目標 (3個月內)  
- 實現真正的「全方位陪伴聊天機器人」
- 保持 Rushia 角色魅力的同時，具備豐富的對話能力
- 用戶滿意度和互動深度顯著提升

## ⚠️ 注意事項

1. **平衡性原則**: 多樣性不能犧牲角色一致性
2. **漸進式實施**: 分階段推進，及時調整策略
3. **用戶反饋**: 持續收集用戶意見，優化話題選擇
4. **內容品質**: 新增內容必須符合 Rushia 的世界觀和價值觀

這個計劃將幫助你的 AI 從「單一角色聊天機器人」進化為「多樣化智能陪伴」，同時保持 Rushia 獨特的魅力！
