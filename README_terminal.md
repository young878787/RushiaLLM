# VTuber AI LLM 終端版本

本地自訂化的LLM終端交互系統，專為VTuber AI應用設計，基於Windows 11優化。

## 特色功能
- 🚀 Qwen-8B 主模型 (4bit量化，極致記憶體優化)
- 🔍 Qwen3-Embedding-0.6B 嵌入模型 (8bit量化)
- 📚 RAG (檢索增強生成) 流程
- 🎭 VTuber 角色扮演優化
- 💬 彩色終端交互界面
- ⚡ Windows 11 性能優化

## 系統需求
- Windows 11
- Python 3.9+
- NVIDIA GPU (建議12GB+ VRAM)
- 16GB+ RAM

## 快速開始

### 1. 準備模型文件
請將以下模型文件放置到對應目錄：
- `models/Qwen-8B/` - 主要對話模型
- `models/Qwen3-Embedding-0.6B/` - 嵌入模型

### 2. 安裝依賴
```bash
# 方法1: 使用批處理腳本 (推薦)
start_terminal.bat

# 方法2: 手動安裝
pip install -r requirements.txt
python main.py
```

## 使用說明

### 基本對話
直接輸入任何文字與AI助手對話，系統會自動使用RAG檢索相關知識。

### 可用指令
- `/help` - 顯示幫助信息
- `/add <文件路徑>` - 添加文檔到知識庫
  - 支持格式: .txt, .pdf, .docx
  - 例如: `/add scrpitsV2/LLM/data/documents/manual.pdf`
- `/search <關鍵詞>` - 搜索知識庫內容
- `/stats` - 顯示系統統計信息
- `/clear` - 清空知識庫
- `/rag on/off` - 啟用/禁用RAG檢索
- `/exit` - 退出程序

### 快捷鍵
- `Ctrl+C` - 安全退出程序

## 配置說明

主要配置文件：`config.yaml`

### VTuber 角色設定
```yaml
vtuber:
  character:
    name: "露西亞"
    personality: "友善、活潑、樂於助人"
    speaking_style: "輕鬆自然，偶爾使用可愛的語氣詞"
```

### 模型配置
```yaml
models:
  llm:
    model_path: "models/Qwen-8B"
    device: "cuda"
    quantization: "4bit"
```

### RAG 設定
```yaml
rag:
  retrieval:
    top_k: 5
    similarity_threshold: 0.7
    chunk_size: 1000
```

## 目錄結構
```
├── main.py                 # 主程序 (終端版本)
├── config.yaml            # 配置文件
├── requirements.txt       # 依賴包
├── start_terminal.bat     # 啟動腳本
├── src/                   # 核心模組
│   ├── llm_manager.py     # LLM 管理器 生成控制
│   ├── rag_system.py      # RAG 系統
│   ├── core.py            # 人格核心
│   ├── filters.py         # 過濾器模組
│   └── utils/             # 工具模組 優化類 主要日誌輸出
├── models/                # 模型文件目錄
└── scrpitsV2/LLM/         # 數據存儲目錄
    ├── data/              # 數據目錄
    │   ├── documents/     # 文檔存放
    │   └── vectordb/      # 向量數據庫
    ├── cache/             # 緩存目錄
    └── logs/              # 日誌目錄
```

## 故障排除

### 常見問題
1. **模型載入失敗**
   - 檢查模型文件是否完整
   - 確認GPU記憶體是否足夠
   - 嘗試使用CPU模式 (修改config.yaml中的device為"cpu")

2. **依賴安裝失敗**
   - 確保Python版本為3.9+
   - 嘗試升級pip: `python -m pip install --upgrade pip`
   - 使用清華源: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

3. **記憶體不足**
   - 調整config.yaml中的gpu_memory_fraction
   - 關閉其他占用GPU的程序
   - 考慮使用更小的模型

### 性能優化
- 確保使用SSD存儲
- 關閉不必要的後台程序
- 調整Windows電源計劃為高性能模式

## 更新日誌
- v1.0.0: 初始終端版本發布
  - 支持Qwen-8B模型
  - 集成RAG檢索系統
  - 彩色終端界面
  - Windows 11優化

## 授權
本項目僅供學習和研究使用。