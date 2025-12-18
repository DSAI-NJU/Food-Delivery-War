# log_manager.py
import os
import logging
from datetime import datetime
from typing import Optional

class LogManager:
    def __init__(self, app_name: str = "MediaCrawler"):
        self.app_name = app_name
        self.log_dir = "logs"
        self.log_file_path: Optional[str] = None
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.log_dir, f"{self.app_name}_{timestamp}.txt")
        
        # 配置 root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除现有 handlers
        logger.handlers.clear()
        
        # 文件 handler
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.INFO)
        
        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加 handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return self.log_file_path
    
    def log_success(self, message: str):
        """记录成功信息"""
        logging.info(f"✅ {message}")
    
    def log_error(self, message: str, error: Exception = None):
        """记录错误信息"""
        if error:
            logging.error(f"❌ {message}: {error}")
        else:
            logging.error(f"❌ {message}")
    
    def log_warning(self, message: str):
        """记录警告信息"""
        logging.warning(f"⚠️  {message}")
    
    def log_info(self, message: str):
        """记录普通信息"""
        logging.info(f"📝 {message}")
    
    def log_debug(self, message: str):
        """记录调试信息"""
        logging.debug(f"🔍 {message}")
    
    def log_crawler_start(self, platform: str, crawler_type: str, target: str = ""):
        """记录爬虫开始信息"""
        message = f"🚀 开始爬取 - 平台: {platform}, 类型: {crawler_type}"
        if target:
            message += f", 目标: {target}"
        logging.info(message)
    
    def log_crawler_end(self, stats: dict = None):
        """记录爬虫结束信息"""
        if stats:
            message = f"🎉 爬取完成! "
            for key, value in stats.items():
                message += f"{key}: {value}, "
            message = message.rstrip(", ")
        else:
            message = "🎉 爬取完成!"
        logging.info(message)
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return self.log_file_path
    
    def save_custom_log(self, filename: str, content: str, directory: str = "logs"):
        """保存自定义日志文件"""
        # 确保目录存在
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, filename)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {content}\n")
        
        return filepath

# 创建全局实例
log_manager = LogManager()