"""
日誌配置模組
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import colorama
from colorama import Fore, Style

colorama.init()


class ColoredFormatter(logging.Formatter):
    """彩色日誌格式化器"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(log_dir: str = "D:\\RushiaModeV2\\scrpitsV2\\LLM\\logs"):
    """設置日誌系統"""
    
    # 🔥 統一：直接使用傳入的log_dir路徑（已經是絕對路徑）
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)


    # 日誌文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_path / f"vtuber_ai_{timestamp}.log"
    
    # 根日誌器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除現有處理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台處理器（彩色輸出）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # 文件處理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 添加處理器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # 設置第三方庫的日誌級別
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    
    logging.info(f"日誌系統已初始化，日誌文件: {log_file}")