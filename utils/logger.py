"""
Logging utility for Text Rewriting with RLHF
Centralized logging configuration
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

def setup_logger(
    name: str = "text_rewriter",
    log_level: int = logging.INFO,
    log_file: str = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path
        console: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Default logger instance
_logger = None

def get_logger(name: str = "text_rewriter") -> logging.Logger:
    """Get or create logger instance"""
    global _logger
    if _logger is None:
        log_file = os.path.join("logs", f"text_rewriter_{datetime.now().strftime('%Y%m%d')}.log")
        _logger = setup_logger(name, log_file=log_file)
    return _logger

