"""
Main Training System for Text Rewriting with RLHF
Uses Qwen LLM with PPO to update model weights based on rewards
"""

import sys
import os

# Windows-specific memory optimizations - MUST be before torch import
if sys.platform == "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import gc
import time
from typing import Dict, List, Tuple, Optional
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# Import utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from utils.logger import get_logger
    from utils.memory_utils import cleanup_memory, monitor_memory
    from config import training_config, ppo_config, model_config
except ImportError:
    # Fallback
    import logging
    def get_logger(name="trainer"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def cleanup_memory(device="auto", verbose=True):
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    training_config = type('Config', (), {
        'batch_size': 8, 'update_frequency': 5, 'save_frequency': 10,
        'memory_cleanup_frequency': 5, 'max_memory_gb': 7.0
    })()
    ppo_config = type('PPOConfig', (), {
        'learning_rate': 1e-5, 'clip_ratio': 0.2, 'value_coef': 0.1
    })()
    model_config = type('ModelConfig', (), {'max_new_tokens': 96})()

# Windows-specific PyTorch thread settings - with error handling
if sys.platform == "win32":
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass  # Already set
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Already set

from sentiment_analyzer import SentimentAnalyzer
from qwen_rewriter import QwenRewriter, MODEL_NAME, PROMPT
from reward_function import RewardFunction
from model_manager import ModelManager
from transformers import AutoTokenizer
try:
    from trl.experimental.ppo import AutoModelForCausalLMWithValueHead
except ImportError:
    from trl.models import AutoModelForCausalLMWithValueHead

class TextRewritingTrainer:
    """
    Main training system for text rewriting with RLHF
    """
    def __init__(
        self,
        device: str = "auto",
        model_save_path: str = "./models",
        max_memory_gb: float = 7.0,  # Leave 1GB for system
        existing_model=None,  # Share existing Qwen model to avoid reloading
        existing_tokenizer=None  # Share existing tokenizer
    ):
        """
        Initialize the training system
        
        Args:
            device: Device to run on
            model_save_path: Base path to save models (will create timestamped subdirectories)
            max_memory_gb: Maximum memory usage in GB
            existing_model: Existing Qwen model instance to share (saves memory)
            existing_tokenizer: Existing tokenizer instance to share
        """
        self.device = self._get_device(device)
        self.max_memory_gb = max_memory_gb
        self.logger = get_logger("TextRewritingTrainer")
        
        self.logger.info(f"Initializing training system on {self.device}")
        self.logger.info(f"Memory limit: {max_memory_gb} GB")
        self.logger.info("Using Qwen LLM with PPO to update model weights")
        
        # Initialize components - share Qwen model to save memory
        self.sentiment_analyzer = SentimentAnalyzer(device=self.device)
        self.reward_function = RewardFunction(self.sentiment_analyzer)
        self.model_manager = ModelManager(model_save_path)
        
        # Base path for models - will create timestamped subdirectories when saving
        self.base_model_save_path = model_save_path
        os.makedirs(self.base_model_save_path, exist_ok=True)
        
        # Current model save path (will be set when saving)
        self.model_save_path = None
        
        # Load the model with value head for PPO
        # First check if there's a latest model in the base directory
        latest_model_path = self.model_manager.get_latest_model("qwen_rewriter")
        
        # Check if a trained model exists (look for config.json, model.safetensors, or pytorch_model.bin)
        model_exists = False
        load_path = MODEL_NAME
        
        if latest_model_path:
            # latest_model_path could be either:
            # 1. Directory path (new format)
            # 2. File path to model.pt (old format in metadata)
            if os.path.isdir(latest_model_path):
                model_dir = latest_model_path
            elif os.path.isfile(latest_model_path):
                # Old format - extract directory
                model_dir = os.path.dirname(latest_model_path)
            else:
                # Try parsing - check if it ends with model.pt
                if latest_model_path.endswith('model.pt'):
                    model_dir = os.path.dirname(latest_model_path)
                else:
                    model_dir = latest_model_path
            
            # Check if the directory has HuggingFace model files
            if os.path.exists(model_dir):
                config_exists = os.path.exists(os.path.join(model_dir, "config.json"))
                model_files_exist = (
                    os.path.exists(os.path.join(model_dir, "model.safetensors")) or
                    os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
                )
                
                if config_exists and model_files_exist:
                    model_exists = True
                    load_path = model_dir
                    self.model_save_path = model_dir  # Use existing model path
                    self.logger.info(f"Found existing model directory: {model_dir}")
                else:
                    self.logger.warning(f"Model directory exists but missing model files: {model_dir}")
        
        if model_exists and self.model_save_path:
            self.logger.info(f"Loading existing trained model from {self.model_save_path}")
        else:
            self.logger.info(f"No existing model found, loading base model {MODEL_NAME}")
            self.model_save_path = None  # Will be created on first save
        
        # Reuse existing tokenizer if available to save memory
        if existing_tokenizer is not None:
            print("Reusing existing tokenizer to save memory...")
            self.tokenizer = existing_tokenizer
        else:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Always use float32 for Windows stability (float16 can cause memory issues)
        model_dtype = torch.float32
        
        # Cleanup before loading - especially important if we're unloading an existing model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # If we have an existing model, we can try to free it before loading
        if existing_model is not None:
            print("Freeing existing model memory before loading trainer model...")
            del existing_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        try:
            # Try with safetensors first, fallback if not available
            try:
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    load_path, 
                    dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True  # Better Windows compatibility
                )
                self.logger.info("Model loaded successfully with safetensors")
            except Exception as e1:
                self.logger.warning(f"Failed to load with safetensors: {e1}, trying without safetensors...")
                # Fallback without safetensors
                try:
                    self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                        load_path, 
                        dtype=model_dtype,
                        low_cpu_mem_usage=True,
                        use_safetensors=False
                    )
                    self.logger.info("Model loaded successfully without safetensors")
                except Exception as e2:
                    self.logger.error(f"Failed to load model from {load_path}: {e2}")
                    raise RuntimeError(f"Could not load model from {load_path}. Original error: {e2}") from e2
            
            self.model.to(self.device)
            
            # Ensure model is properly initialized - test forward pass
            with torch.no_grad():
                dummy_input = self.tokenizer("test", return_tensors="pt").to(self.device)
                if dummy_input['input_ids'].numel() > 0:
                    _ = self.model(**dummy_input)
                    del dummy_input
                gc.collect()
            
            self.logger.info(f"Model initialized successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        # Create QwenRewriter WITHOUT loading a new model - share the existing one
        # This avoids duplicate memory allocation (prevents Windows memory errors)
        from qwen_rewriter import QwenRewriter
        try:
            self.qwen_rewriter = QwenRewriter.__new__(QwenRewriter)  # Create without __init__
            self.qwen_rewriter.device = self.device
            self.qwen_rewriter.max_new_tokens = 96
            self.qwen_rewriter.tokenizer = self.tokenizer
            # Share the underlying pretrained model (not the value head wrapper)
            self.qwen_rewriter.model = self.model.pretrained_model
            
            # Ensure the shared model is in eval mode
            self.qwen_rewriter.model.eval()
        except Exception as e:
            print(f"Warning: Could not share model instance: {e}")
            # Fallback: create normally but this uses more memory
            self.qwen_rewriter = QwenRewriter(device=self.device)
        
        # Use custom PPO implementation instead of trl's PPOTrainer (more reliable, simpler API)
        # Initialize optimizer for PPO updates
        self.ppo_learning_rate = ppo_config.learning_rate
        self.ppo_cliprange = ppo_config.clip_ratio
        self.ppo_vf_coef = ppo_config.value_coef
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.ppo_learning_rate)
        
        # Initial memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Training configuration from config module
        self.training_config = {
            'max_episodes': training_config.max_episodes,
            'min_reward_threshold': training_config.min_reward_threshold,
            'patience': training_config.patience,
            'batch_size': training_config.batch_size,
            'update_frequency': training_config.update_frequency,
            'save_frequency': training_config.save_frequency,
            'memory_cleanup_frequency': training_config.memory_cleanup_frequency
        }
        
        # Training state
        self.training_state = {
            'episode': 0,
            'best_reward': 0.0,
            'patience_counter': 0,
            'recent_rewards': [],
            'training_history': []
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _monitor_memory(self):
        """Monitor and manage memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            
            if allocated > self.max_memory_gb * 0.8:  # 80% threshold
                print(f"High memory usage: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
                self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Clean up memory aggressively"""
        cleanup_memory(self.device, verbose=False)
    
    def _analyze_input_text(self, text: str) -> Dict:
        """Analyze input text and determine if rewriting is needed"""
        analysis = self.sentiment_analyzer.analyze_text(text)
        
        # Determine if text needs improvement
        needs_improvement = (
            analysis['sentiment_score'] < 0.2 or
            analysis['tone_score'] < 0.2 or
            analysis['overall_score'] < 0.5
        )
        
        return {
            'analysis': analysis,
            'needs_improvement': needs_improvement,
            'target_style': self._determine_target_style(analysis)
        }
    
    def _determine_target_style(self, analysis: Dict) -> str:
        """Determine target style based on analysis"""
        if analysis['sentiment_score'] < -0.3:
            return "positive"
        elif analysis['tone_score'] < -0.3:
            return "polite"
        else:
            return "friendly"
    
    def _generate_training_experience(self, text: str) -> Dict:
        """Generate training experience for a single text"""
        # Analyze input
        input_analysis = self._analyze_input_text(text)
        
        if not input_analysis['needs_improvement']:
            return None
        
        # Generate rewritten text using Qwen LLM
        # Use the PPO-trained model for generation
        rewritten_text = self.qwen_rewriter.rewrite(text)
        
        # Ensure qwen_rewriter is using the latest model
        if hasattr(self, 'model') and self.qwen_rewriter.model != self.model.pretrained_model:
            self.qwen_rewriter.model = self.model.pretrained_model
            self.qwen_rewriter.tokenizer = self.tokenizer
        
        # Compute reward
        reward, component_scores = self.reward_function.compute_reward(text, rewritten_text)
        
        # Prepare experience data
        experience = {
            'original_text': text,
            'rewritten_text': rewritten_text,
            'target_style': input_analysis['target_style'],
            'reward': reward,
            'component_scores': component_scores,
            'input_analysis': input_analysis['analysis']
        }
        
        return experience
    
    def _manual_ppo_step(self, query_ids, response_ids, reward):
        """Manual PPO update step (alternative to trl's PPOTrainer)"""
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # Ensure query_ids and response_ids are 2D tensors [batch_size, seq_len]
            if query_ids.dim() == 1:
                query_ids = query_ids.unsqueeze(0)
            if response_ids.dim() == 1:
                response_ids = response_ids.unsqueeze(0)
            
            # Concatenate query and response along sequence dimension
            # query_ids shape: [1, query_len], response_ids shape: [1, resp_len]
            input_ids = torch.cat([query_ids, response_ids], dim=1)  # Result: [1, query_len + resp_len]
            
            # Create attention mask (all ones for valid tokens)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            
            # Get model outputs - AutoModelForCausalLMWithValueHead returns value directly
            # Use use_cache=False to avoid issues with cached positional embeddings
            # Request hidden states for value head computation
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                return_dict=True,
                output_hidden_states=True,  # Request hidden states for value head
                use_cache=False  # Disable cache to avoid positional embedding issues
            )
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Get value predictions from value head - FIXED IMPLEMENTATION
            # AutoModelForCausalLMWithValueHead should return value in outputs
            values = None
            
            # Method 1: Check if value is directly in outputs
            if hasattr(outputs, 'value') and outputs.value is not None:
                values = outputs.value
                if values.dim() > 1:
                    values = values.squeeze(-1)
            
            # Method 2: Extract from hidden states and compute via v_head
            if values is None and hasattr(self.model, 'v_head'):
                # Try to get last hidden state from outputs
                last_hidden = None
                
                # Try hidden_states first (most reliable)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    if isinstance(outputs.hidden_states, tuple) and len(outputs.hidden_states) > 0:
                        # Get last layer's last token
                        last_layer = outputs.hidden_states[-1]  # [batch, seq, hidden]
                        if last_layer.dim() == 3:
                            last_hidden = last_layer[:, -1, :]  # [batch, hidden]
                
                # Fallback to last_hidden_state
                if last_hidden is None and hasattr(outputs, 'last_hidden_state'):
                    last_hidden_state = outputs.last_hidden_state
                    if last_hidden_state.dim() == 3:
                        last_hidden = last_hidden_state[:, -1, :]
                
                # Last resort: get from pretrained model
                if last_hidden is None:
                    with torch.no_grad():
                        try:
                            base_outputs = self.model.pretrained_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                                use_cache=False
                            )
                            if hasattr(base_outputs, 'hidden_states') and base_outputs.hidden_states:
                                if isinstance(base_outputs.hidden_states, tuple) and len(base_outputs.hidden_states) > 0:
                                    last_hidden = base_outputs.hidden_states[-1][:, -1, :]
                            elif hasattr(base_outputs, 'last_hidden_state'):
                                last_hidden_state = base_outputs.last_hidden_state
                                if last_hidden_state.dim() == 3:
                                    last_hidden = last_hidden_state[:, -1, :]
                        except Exception as e:
                            print(f"Warning: Could not extract hidden state: {e}")
                
                # Compute value from hidden state
                if last_hidden is not None:
                    # Ensure correct shape and dimensions
                    if last_hidden.dim() == 1:
                        last_hidden = last_hidden.unsqueeze(0)
                    
                    # Get hidden dimension from v_head
                    try:
                        if hasattr(self.model.v_head, 'summary'):
                            # ValueHead typically has a summary layer
                            hidden_dim = self.model.v_head.summary.in_features
                        elif hasattr(self.model.v_head, 'linear'):
                            hidden_dim = self.model.v_head.linear.in_features
                        else:
                            # Default for Qwen2.5-0.5B
                            hidden_dim = last_hidden.shape[-1]
                        
                        # Ensure correct dimension
                        if last_hidden.shape[-1] != hidden_dim:
                            # If dimension mismatch, try to adjust
                            if last_hidden.shape[-1] > hidden_dim:
                                last_hidden = last_hidden[:, :hidden_dim]
                            else:
                                # Pad if needed
                                pad_size = hidden_dim - last_hidden.shape[-1]
                                last_hidden = torch.nn.functional.pad(
                                    last_hidden, (0, pad_size), mode='constant', value=0
                                )
                        
                        # Compute value
                        values = self.model.v_head(last_hidden)
                        if values.dim() > 1:
                            values = values.squeeze(-1)
                    except Exception as e:
                        print(f"Warning: Error computing value from v_head: {e}")
                        values = torch.zeros(1, device=self.device, dtype=torch.float32)
            
            # Final fallback: zero value
            if values is None:
                values = torch.zeros(1, device=self.device, dtype=torch.float32)
                print("Warning: Using zero value as fallback")
            
            # Compute log probabilities for the response tokens only
            # The logits are shifted by 1 (each position predicts next token)
            query_len = query_ids.shape[1]
            response_len = response_ids.shape[1]
            
            # Extract logits corresponding to response tokens
            # Logits at position i predict token at position i+1
            # So for response starting at query_len, we need logits from query_len-1 to query_len+response_len-2
            response_logits_start = max(0, query_len - 1)
            response_logits_end = query_len + response_len - 1
            
            if response_logits_end > logits.shape[1]:
                response_logits_end = logits.shape[1]
            
            response_logits = logits[:, response_logits_start:response_logits_end, :]
            
            # Align response tokens with their corresponding logits
            # If response starts at query_len, tokens are at positions query_len to query_len+response_len-1
            # But logits are shifted, so we use tokens starting from query_len
            actual_response_tokens = input_ids[:, query_len:query_len+response_len]
            
            # Ensure shapes match
            if response_logits.shape[1] != actual_response_tokens.shape[1]:
                min_len = min(response_logits.shape[1], actual_response_tokens.shape[1])
                response_logits = response_logits[:, :min_len, :]
                actual_response_tokens = actual_response_tokens[:, :min_len]
            
            # Compute log probs
            log_probs = F.log_softmax(response_logits, dim=-1)
            response_log_probs = torch.gather(
                log_probs.reshape(-1, log_probs.shape[-1]), 
                1, 
                actual_response_tokens.reshape(-1, 1)
            ).reshape(actual_response_tokens.shape)
            
            # Compute advantage (simple version: reward - value)
            advantage = reward - values.squeeze()
            
            # PPO loss: maximize (advantage * log_prob) with clipping
            # Simplified version: direct policy gradient with advantage weighting
            policy_loss = -(advantage * response_log_probs.mean()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), reward)
            
            # Total loss
            total_loss = policy_loss + self.ppo_vf_coef * value_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            self.model.eval()
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss.item(),
                'reward': reward.item(),
                'advantage': advantage.item()
            }
        except Exception as e:
            self.model.eval()
            self.logger.error(f"Error in manual PPO step: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _update_policy(self, experiences: List[Dict]):
        """Update Qwen model weights using PPO"""
        if not experiences:
            return {}
        
        # Update Qwen model for each experience
        stats_list = []
        
        for exp in experiences:
            # Build prompt
            query = PROMPT.format(text=exp['original_text'].strip())
            
            # Tokenize prompt
            enc = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            input_ids = enc["input_ids"]
            input_len = input_ids.shape[1]
            
            # Use the user-selected rewritten text (not regenerate)
            # Tokenize the selected rewritten text
            selected_text = exp['rewritten_text']
            
            # Safety check for valid text
            if not selected_text or len(selected_text.strip()) == 0:
                print(f"Warning: Empty rewritten text, skipping experience")
                continue
            
            try:
                selected_tokens = self.tokenizer(
                    selected_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128,
                    add_special_tokens=False
                ).to(self.device)
                
                # Use selected tokens as response
                resp_only = selected_tokens['input_ids']
                
                # Validate token tensor
                if resp_only.numel() == 0 or resp_only is None:
                    raise ValueError("Empty tokens")
            except Exception as e:
                print(f"Warning: Error tokenizing selected text: {e}, falling back to generation")
                # Fallback to generating
                with torch.no_grad():
                    generated = self.model.generate(
                        **enc,
                        do_sample=True,
                        max_new_tokens=96,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                resp_only = generated[:, input_len:]
                if resp_only.numel() == 0:
                    resp_only = generated[:, -1:]
            
            # Convert reward to tensor
            reward_tensor = torch.tensor([exp['reward']], dtype=torch.float32).to(self.device)
            
            # Update model with PPO - with safety checks
            try:
                # Validate tensors before PPO step
                if input_ids.numel() == 0 or resp_only.numel() == 0:
                    print(f"Warning: Empty tensors, skipping PPO update")
                    continue
                
                # Ensure tensors are on correct device and contiguous
                input_ids = input_ids.contiguous().to(self.device)
                resp_only = resp_only.contiguous().to(self.device)
                reward_tensor = reward_tensor.to(self.device)
                
                # Clear cache before PPO step
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Manual PPO update (simpler and more reliable than trl's PPOTrainer)
                stats = self._manual_ppo_step(input_ids[0], resp_only[0], reward_tensor[0])
                stats_list.append(stats)
                
                # Cleanup after each PPO step
                del input_ids, resp_only, reward_tensor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Error in PPO step: {e}")
                import traceback
                traceback.print_exc()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                continue
        
        # Average stats if any
        avg_stats = {}
        if stats_list:
            for key in stats_list[0].keys():
                if isinstance(stats_list[0][key], (int, float)):
                    avg_stats[key] = np.mean([s[key] for s in stats_list if key in s])
        
        # Save model after PPO update with proper metadata
        if stats_list and not any('error' in s for s in stats_list):
            try:
                self.save_model_after_update(experiences, avg_stats)
            except Exception as e:
                self.logger.error(f"Error saving model after update: {e}", exc_info=True)
        
        return avg_stats
    
    def save_model_after_update(self, experiences: List[Dict], update_stats: Dict = None):
        """
        Save model after PPO update with proper timestamped directory and metadata
        
        Args:
            experiences: List of experiences used for update
            update_stats: Statistics from PPO update
        """
        from datetime import datetime
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.base_model_save_path, f"qwen_rewriter_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        self.logger.info(f"Saving updated model to {model_dir}")
        
        # Set model to eval mode before saving
        self.model.eval()
        
        # Clean up memory before saving
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            # Save model and tokenizer
            self.model.save_pretrained(model_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(model_dir)
            
            # Prepare metadata
            training_stats = {
                'reward': experiences[0]['reward'] if experiences else 0.0,
                'components': experiences[0].get('component_scores', {}) if experiences else {},
                'iterations': len(experiences),
                'timestamp': datetime.now().isoformat()
            }
            
            if update_stats:
                training_stats['ppo_stats'] = update_stats
            
            metadata = {
                'model_name': 'qwen_rewriter',
                'timestamp': timestamp,
                'model_path': model_dir,  # Directory path where HuggingFace model is saved
                'training_stats': training_stats
            }
            
            # Save metadata
            metadata_path = os.path.join(model_dir, "metadata.json")
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update current model save path
            self.model_save_path = model_dir
            
            self.logger.info(f"Model saved successfully to {model_dir}")
            print(f"✓ Model saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}", exc_info=True)
            print(f"⚠️ Warning: Error saving model: {e}")
            raise
    
    def train_episode(self, input_text: str) -> Dict:
        """Train for one episode with a single input text"""
        print(f"\n--- Episode {self.training_state['episode'] + 1} ---")
        print(f"Input text: {input_text}")
        
        # Generate experience
        experience = self._generate_training_experience(input_text)
        
        if experience is None:
            print("Text doesn't need improvement, skipping episode")
            return {'reward': 0.0, 'skipped': True}
        
        print(f"Rewritten text: {experience['rewritten_text']}")
        print(f"Reward: {experience['reward']:.3f}")
        
        # Update training state
        self.training_state['recent_rewards'].append(experience['reward'])
        self.training_state['training_history'].append(experience)
        
        # Update best reward
        if experience['reward'] > self.training_state['best_reward']:
            self.training_state['best_reward'] = experience['reward']
            self.training_state['patience_counter'] = 0
        else:
            self.training_state['patience_counter'] += 1
        
        # Update Qwen model weights if enough experiences
        if len(self.training_state['training_history']) >= self.training_config['update_frequency']:
            print("Updating Qwen model weights with PPO...")
            update_stats = self._update_policy(self.training_state['training_history'])
            self.training_state['training_history'] = []  # Clear history
            
            # Model is already saved in _update_policy via save_model_after_update
            # Update qwen_rewriter to use the updated model
            self.qwen_rewriter.model = self.model.pretrained_model
            self.qwen_rewriter.model.eval()  # Set to eval mode for inference
            self.qwen_rewriter.tokenizer = self.tokenizer
            
            if update_stats:
                print(f"PPO update stats: {update_stats}")
        
        # Memory management
        self._monitor_memory()
        
        if self.training_state['episode'] % self.training_config['memory_cleanup_frequency'] == 0:
            self._cleanup_memory()
        
        self.training_state['episode'] += 1
        
        return {
            'reward': experience['reward'],
            'component_scores': experience['component_scores'],
            'skipped': False
        }
    
    def should_stop_training(self) -> bool:
        """Determine if training should stop"""
        # Check if max episodes reached
        if self.training_state['episode'] >= self.training_config['max_episodes']:
            print("Maximum episodes reached")
            return True
        
        # Check if reward threshold met
        if len(self.training_state['recent_rewards']) >= 5:
            recent_avg = np.mean(self.training_state['recent_rewards'][-5:])
            if recent_avg >= self.training_config['min_reward_threshold']:
                print(f"Reward threshold met: {recent_avg:.3f}")
                return True
        
        # Check patience
        if self.training_state['patience_counter'] >= self.training_config['patience']:
            print("Patience exceeded, stopping training")
            return True
        
        return False
    
    def save_model(self, episode: int):
        """Save the current Qwen model with episode metadata"""
        from datetime import datetime
        
        # Create timestamped directory if not already set
        if not self.model_save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(self.base_model_save_path, f"qwen_rewriter_{timestamp}")
        else:
            model_dir = self.model_save_path
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Set model to eval mode before saving
        self.model.eval()
        
        # Clean up memory before saving
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            # Save Qwen model with updated weights
            self.model.save_pretrained(model_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(model_dir)
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata = {
                'model_name': 'qwen_rewriter',
                'timestamp': timestamp_str,
                'model_path': model_dir,  # Directory path where HuggingFace model is saved
                'episode': episode,
                'best_reward': self.training_state['best_reward'],
                'recent_rewards': self.training_state['recent_rewards'][-10:],  # Last 10 rewards
                'training_config': self.training_config,
                'training_stats': {
                    'episode': episode,
                    'best_reward': self.training_state['best_reward'],
                    'recent_rewards': self.training_state['recent_rewards'][-10:],
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Save metadata
            import json
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.model_save_path = model_dir
            print(f"Qwen model saved at episode {episode} to {model_dir}")
            return model_dir
        except Exception as e:
            self.logger.error(f"Error saving model: {e}", exc_info=True)
            raise
    
    def load_model(self, model_path: str):
        """Load a saved Qwen model"""
        try:
            # Reload Qwen model from saved path
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_path,
                dtype=torch.float32
            )
            self.model.to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Recreate optimizer with updated model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.ppo_learning_rate)
        
            # Update qwen_rewriter to use the loaded model
            self.qwen_rewriter.model = self.model.pretrained_model
            self.qwen_rewriter.model.eval()  # Set to eval mode for inference
            self.qwen_rewriter.tokenizer = self.tokenizer
            
            print(f"Qwen model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def train(self, input_texts: List[str], resume_from: Optional[str] = None):
        """
        Main training loop
        
        Args:
            input_texts: List of input texts to train on
            resume_from: Optional path to resume from
        """
        print("Starting training...")
        
        # Load model if resuming
        if resume_from:
            if not self.load_model(resume_from):
                print("Failed to load model, starting fresh")
        
        # Training loop
        text_index = 0
        
        while not self.should_stop_training():
            # Get next input text
            if text_index >= len(input_texts):
                text_index = 0  # Cycle through texts
            
            input_text = input_texts[text_index]
            text_index += 1
            
            # Train episode
            episode_result = self.train_episode(input_text)
            
            # Save model periodically
            if (self.training_state['episode'] % self.training_config['save_frequency'] == 0 and 
                self.training_state['episode'] > 0):
                self.save_model(self.training_state['episode'])
        
        # Final save
        final_model_path = self.save_model(self.training_state['episode'])
        
        print(f"\nTraining completed!")
        print(f"Final episode: {self.training_state['episode']}")
        print(f"Best reward: {self.training_state['best_reward']:.3f}")
        print(f"Final model saved to: {final_model_path}")
        
        return final_model_path
    
    def get_training_summary(self) -> str:
        """Get training summary"""
        summary = f"""
Training Summary:
- Episodes completed: {self.training_state['episode']}
- Best reward: {self.training_state['best_reward']:.3f}
- Recent rewards: {self.training_state['recent_rewards'][-5:] if self.training_state['recent_rewards'] else 'None'}
- Patience counter: {self.training_state['patience_counter']}
        """.strip()
        
        return summary
