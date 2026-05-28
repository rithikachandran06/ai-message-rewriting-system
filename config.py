"""
Configuration management for Text Rewriting with RLHF
All configuration values centralized here
"""

from dataclasses import dataclass
from typing import Dict
import os

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    max_episodes: int = 100
    min_reward_threshold: float = 0.8
    patience: int = 10
    batch_size: int = 8
    update_frequency: int = 5
    save_frequency: int = 10
    memory_cleanup_frequency: int = 5
    max_memory_gb: float = 7.0
    
@dataclass
class PPOConfig:
    """PPO algorithm configuration"""
    learning_rate: float = 1e-5
    clip_ratio: float = 0.2
    value_coef: float = 0.1
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    ppo_epochs: int = 1
    
@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    max_length: int = 512
    model_save_path: str = "./models"
    
@dataclass
class RewardConfig:
    """Reward function configuration"""
    weights: Dict[str, float] = None
    quality_bonus_weight: float = 0.1
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'user_rating': 0.4,
                'sentiment_improvement': 0.2,
                'tone_improvement': 0.15,
                'meaning_preservation': 0.15,
                'intent_preservation': 0.1
            }
        # Validate weights sum to ~1.0 (including quality_bonus_weight)
        # Note: quality_bonus_weight is applied separately, so main weights should sum to 0.9
        main_weights_sum = sum(self.weights.values())
        total_with_bonus = main_weights_sum + self.quality_bonus_weight
        if abs(total_with_bonus - 1.0) > 0.01:
            # Auto-adjust to ensure it sums to 1.0
            if total_with_bonus > 1.0:
                # Scale down main weights proportionally
                scale_factor = (1.0 - self.quality_bonus_weight) / main_weights_sum
                self.weights = {k: v * scale_factor for k, v in self.weights.items()}
            # Allow small tolerance for floating point errors
            final_sum = sum(self.weights.values()) + self.quality_bonus_weight
            if abs(final_sum - 1.0) > 0.01:
                raise ValueError(f"Reward weights sum to {final_sum}, should be ~1.0")

@dataclass
class AppConfig:
    """Main application configuration"""
    device: str = "auto"
    low_memory: bool = True
    reward_threshold: float = 0.75
    max_iterations: int = 5
    
    # Windows-specific settings
    enable_windows_optimizations: bool = True
    pytorch_threads: int = 1

# Constants
class Constants:
    """Application-wide constants"""
    # Temperature values for multiple version generation
    TEMPERATURES = [0.5, 0.7, 0.9]
    
    # Rating mappings
    RATING_MAP = {
        '1': 0.0,
        '2': 0.25,
        '3': 0.5,
        '4': 0.75,
        '5': 1.0
    }
    
    # Stop words for meaning preservation
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
        'been', 'have', 'has', 'had', 'do', 'does', 'did'
    }
    
    # Quality thresholds
    MIN_SENTIMENT_SCORE = 0.3
    MIN_TONE_SCORE = 0.3
    MIN_MEANING_SCORE = 0.7
    MIN_INTENT_SCORE = 0.5
    MEANING_PRESERVATION_THRESHOLD = 0.6

# Global configuration instances
training_config = TrainingConfig()
ppo_config = PPOConfig()
model_config = ModelConfig()
reward_config = RewardConfig()
app_config = AppConfig()

