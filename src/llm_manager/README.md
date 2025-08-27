# LLM Manager 重構版 - 使用指南

## 📁 模組結構

```
llm_manager/
├── __init__.py              # 模組入口和公開接口
├── llm_manager.py          # 主要管理器 - 統一接口
├── gpu_manager.py          # GPU資源管理模組
├── model_loader.py         # 模型載入和初始化模組
├── response_generator.py   # 回應生成和處理模組
└── README.md              # 使用指南（本文件）
```

## 🚀 快速開始

### 基本使用

```python
from llm_manager import create_llm_manager

# 創建並初始化LLM管理器
config = {...}  # 你的配置
manager = await create_llm_manager(config)

if manager:
    # 生成回應
    response = await manager.generate_response("你好！")
    print(response)
    
    # 清理資源
    manager.cleanup()
```

### 詳細使用

```python
from llm_manager import LLMManager

# 手動創建管理器
manager = LLMManager(config)

# 初始化所有組件
if await manager.initialize():
    
    # 生成帶上下文的回應
    response = await manager.generate_response(
        prompt="今天天氣怎麼樣？",
        context="今天是晴天，溫度25度",
        conversation_history=[
            ("早上好！", "早上好！今天是美好的一天呢～"),
            ("你今天心情如何？", "心情很好呢！謝謝你的關心～")
        ],
        rag_enabled=True
    )
    
    # 獲取模型信息
    info = manager.get_model_info()
    print(f"GPU使用情況: {info['gpu_cluster']}")
    
    # 獲取對話統計
    stats = manager.get_conversation_stats()
    print(f"對話次數: {stats['conversation_count']}")
    
    # 清理資源
    manager.cleanup()
```

## 🔧 核心功能

### 1. GPU管理

```python
# 獲取GPU狀態
status = manager.get_gpu_status()

# 優化GPU記憶體
manager.optimize_gpu_memory()

# 診斷GPU分配
diagnosis = manager.diagnose_gpu_allocation()
```

### 2. 模型信息

```python
# 獲取完整模型信息
info = manager.get_model_info()

# 訪問具體組件（向後兼容）
llm_model = manager.llm_model
tokenizer = manager.llm_tokenizer
embedding = manager.embedding_model
```

### 3. 嵌入向量

```python
# 獲取文本嵌入
texts = ["你好", "再見"]
embeddings = manager.get_embeddings(texts)
```

### 4. RAG系統整合

```python
# 設置RAG系統引用
manager.set_rag_system_reference(your_rag_system)

# 使用RAG增強的回應生成
response = await manager.generate_response(
    "告訴我關於露西婭的信息", 
    rag_enabled=True
)
```

## 🏗️ 模組化架構優勢

### 1. **職責分離**
- `gpu_manager.py`: 專注GPU資源管理
- `model_loader.py`: 專注模型載入邏輯  
- `response_generator.py`: 專注回應生成處理
- `llm_manager.py`: 提供統一接口

### 2. **易於維護**
- 每個模組功能單一明確
- 問題定位更精準
- 代碼結構清晰

### 3. **良好擴展性**
- 可以獨立優化各個模組
- 容易添加新功能
- 支持插件化架構

### 4. **向後兼容**
- 保持原有API不變
- 現有代碼無需修改
- 平滑遷移

## 📊 系統診斷工具

```python
from llm_manager import get_system_info, diagnose_system

# 獲取系統信息
sys_info = get_system_info()

# 系統診斷
diagnosis = diagnose_system()
print("檢測到的問題:", diagnosis["issues"])
print("建議:", diagnosis["recommendations"])
```

## ⚡ 性能優化

### 多GPU配置
- LLM主模型：自動使用前4張GPU卡
- 嵌入模型：使用第5張GPU或最後一張
- 智能記憶體分配和負載均衡

### 量化優化
- LLM模型：4bit量化 (BitsAndBytesConfig)
- 嵌入模型：8bit量化
- 自動回退到標準模式

### 記憶體管理
- 智能記憶體清理
- OOM自動重試機制
- 動態批次大小調整

## 🔍 調試和監控

### GPU狀態監控
```python
# 即時GPU狀態
status = manager.get_gpu_status()
print(f"GPU數量: {status['gpu_count']}")
print(f"總記憶體: {status['total_memory_gb']:.1f}GB")
print(f"可用記憶體: {status['available_memory_gb']:.1f}GB")
```

### 對話統計
```python
# 對話統計信息
stats = manager.get_conversation_stats()
print(f"對話次數: {stats['conversation_count']}")
print(f"當前心情: {stats['current_mood']}")
print(f"親密度: {stats['intimacy_level']}")
```

## 🚨 錯誤處理

### 初始化失敗處理
```python
manager = await create_llm_manager(config)
if manager is None:
    # 檢查系統診斷
    diagnosis = diagnose_system()
    print("初始化失敗，可能的原因:", diagnosis["issues"])
```

### 記憶體不足處理
```python
try:
    response = await manager.generate_response(prompt)
except torch.cuda.OutOfMemoryError:
    # 自動清理記憶體並重試
    manager.optimize_gpu_memory()
    response = await manager.generate_response(prompt)
```

## 📝 配置文件示例

```yaml
models:
  llm:
    model_path: "/path/to/Qwen-8B"
    device: "cuda"
    max_length: 4096
    temperature: 0.7
    top_p: 0.9
    top_k: 50
  
  embedding:
    model_path: "/path/to/Qwen3-Embedding-0.6B" 
    max_length: 512
    batch_size: 32

vtuber:
  response:
    max_tokens: 150
    min_tokens: 10
    filter_repetition: true
    enable_traditional_chinese: true
```

## 🔄 遷移指南

### 從舊版本遷移

```python
# 舊代碼
from src.llm_manager import LLMManager

# 新代碼 - 無需修改！
from llm_manager import LLMManager
# 所有現有API保持不變
```

### 新功能使用

```python
# 使用新的工廠函數
manager = await create_llm_manager(config)

# 使用新的診斷工具
diagnosis = diagnose_system()

# 使用模組信息
from llm_manager import get_module_info
info = get_module_info()
```
