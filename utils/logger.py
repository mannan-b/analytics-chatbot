# 📝 LOGGER.PY - Logging Configuration

import logging
import sys
from pathlib import Path
from utils.config import config

def setup_logging():
    """Setup logging configuration for the Text-to-SQL system"""
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler
            logging.FileHandler("logs/text_to_sql.log", mode='a')
        ]
    )
    
    # Configure specific loggers
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Logging configured - Level: {config.LOG_LEVEL}")
    
    return logger