"""
Memory management utilities
Centralized memory cleanup and monitoring
"""

import gc
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def cleanup_memory(device: str = "auto", verbose: bool = True):
    """
    Clean up memory aggressively
    
    Args:
        device: Device type ("auto", "cpu", "cuda")
        verbose: Whether to log cleanup actions
    """
    gc.collect()
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"Memory cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def monitor_memory(device: str = "auto", threshold_gb: float = 7.0):
    """
    Monitor memory usage and cleanup if needed
    
    Args:
        device: Device type
        threshold_gb: Memory threshold in GB
        
    Returns:
        Tuple of (allocated_gb, should_cleanup)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        if allocated > threshold_gb * 0.8:  # 80% threshold
            logger.warning(f"High memory usage: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            cleanup_memory(device, verbose=False)
            return allocated, True
        
        return allocated, False
    
    return 0.0, False

def safe_model_unload(model, verbose: bool = True):
    """
    Safely unload a model from memory
    
    Args:
        model: Model to unload
        verbose: Whether to log
    """
    if model is None:
        return
    
    try:
        if hasattr(model, 'cpu'):
            model.cpu()
        del model
        cleanup_memory(verbose=verbose)
        if verbose:
            logger.debug("Model unloaded from memory")
    except Exception as e:
        logger.error(f"Error unloading model: {e}")

class MemoryContext:
    """Context manager for memory cleanup"""
    
    def __init__(self, device: str = "auto", cleanup_on_exit: bool = True):
        self.device = device
        self.cleanup_on_exit = cleanup_on_exit
    
    def __enter__(self):
        cleanup_memory(self.device, verbose=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            cleanup_memory(self.device, verbose=False)
        return False

