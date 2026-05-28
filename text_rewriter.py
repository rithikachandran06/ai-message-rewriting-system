"""
Text Rewriting Module with Policy Network
Rewrites input text to be more polite, positive, and friendly while preserving meaning and intent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    pipeline
)
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import warnings
warnings.filterwarnings("ignore")

class PolicyNetwork(nn.Module):
    """
    Policy network for text rewriting with RLHF
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512):
        super(PolicyNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.1,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        self.value_head = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the policy network
        
        Args:
            input_ids: Token IDs of input text
            attention_mask: Attention mask for padding
            
        Returns:
            Tuple of (logits, value, hidden_states)
        """
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        combined = lstm_out + attn_out
        combined = self.dropout(combined)
        
        # Output projections
        logits = self.output_projection(combined)
        value = self.value_head(combined.mean(dim=1))  # Global average pooling
        
        return logits, value, combined

class TextRewriter:
    """
    Text rewriter that uses a policy network to rewrite text
    """
    def __init__(self, device: str = "auto", model_name: str = "gpt2"):
        """
        Initialize the text rewriter
        
        Args:
            device: Device to run on
            model_name: Base model name for tokenizer
        """
        self.device = self._get_device(device)
        self.model_name = model_name
        
        # Load tokenizer
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = len(self.tokenizer)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
            self.vocab_size = 50257  # GPT-2 vocab size
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(self.vocab_size).to(self.device)
        
        # Load base model for generation (lightweight)
        try:
            self.base_model = GPT2LMHeadModel.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        except Exception as e:
            print(f"Warning: Could not load base model: {e}")
            self.base_model = None
        
        # Rewriting templates and rules
        self.politeness_templates = [
            "Could you please {action}?",
            "I would appreciate it if you could {action}.",
            "Would you mind {action}?",
            "I hope you don't mind, but could you {action}?",
            "If it's not too much trouble, could you {action}?"
        ]
        
        self.positive_phrases = [
            "I'm excited to", "I'm looking forward to", "I appreciate", 
            "Thank you for", "I'm grateful for", "I'm pleased to",
            "I'm happy to", "I'm delighted to", "I'm thrilled to"
        ]
        
        self.friendly_connectors = [
            "I hope this helps!", "Let me know if you need anything else!",
            "Feel free to reach out!", "I'm here to help!",
            "Thanks so much!", "Have a great day!"
        ]
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _extract_action_from_text(self, text: str) -> str:
        """Extract the main action/request from text"""
        # Simple extraction of verbs and objects
        words = text.lower().split()
        
        # Look for common action patterns
        action_patterns = [
            r'(can you|could you|please)\s+(.+?)(?:\?|\.|$)',
            r'(i need|i want|i would like)\s+(.+?)(?:\?|\.|$)',
            r'(help me|show me|tell me)\s+(.+?)(?:\?|\.|$)'
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(2).strip()
        
        # Fallback: return the text without first few words
        if len(words) > 3:
            return ' '.join(words[2:])
        return text
    
    def _apply_rule_based_rewriting(self, text: str, target_style: str = "polite") -> str:
        """
        Apply rule-based rewriting as a baseline
        
        Args:
            text: Input text
            target_style: Target style ("polite", "positive", "friendly")
            
        Returns:
            Rewritten text
        """
        if not text.strip():
            return text
        
        rewritten = text.strip()
        original_lower = rewritten.lower()
        
        # Make polite
        if target_style in ["polite", "positive", "friendly"]:
            # Replace harsh words with polite alternatives
            harsh_replacements = {
                r'\bhate\b': 'dislike',
                r'\bterrible\b': 'not great',
                r'\bawful\b': 'not good',
                r'\bhorrible\b': 'not ideal',
                r'\bstupid\b': 'not the best approach',
                r'\bidiot\b': 'person',
                r'\bdumb\b': 'not clear',
                r'\bannoying\b': 'frustrating',
                r'\bdisgusting\b': 'not appealing',
                r'\bpathetic\b': 'disappointing',
                r'\bworthless\b': 'not useful',
                r'\bpointless\b': 'not necessary'
            }
            
            for harsh, polite in harsh_replacements.items():
                rewritten = re.sub(harsh, polite, rewritten, flags=re.IGNORECASE)
            
            # Add polite language
            if not any(word in rewritten.lower() for word in ['please', 'could you', 'would you', 'may i']):
                if rewritten.endswith('.'):
                    rewritten = rewritten[:-1] + ', please.'
                elif not rewritten.endswith(('!', '?')):
                    rewritten += ', please'
        
        # Make positive
        if target_style in ["positive", "friendly"]:
            # Replace negative phrases with positive ones
            positive_replacements = {
                r'\bi can\'t\b': "I'm having trouble with",
                r'\bi don\'t like\b': "I prefer something different",
                r'\bit\'s bad\b': "it could be better",
                r'\bit\'s wrong\b': "it needs adjustment",
                r'\bthis sucks\b': "this isn't working well",
                r'\bi\'m angry\b': "I'm concerned",
                r'\bi\'m frustrated\b': "I'm having some challenges"
            }
            
            for negative, positive in positive_replacements.items():
                rewritten = re.sub(negative, positive, rewritten, flags=re.IGNORECASE)
            
            # Add positive framing
            if rewritten.startswith('i '):
                rewritten = "I'm happy to " + rewritten[2:]
            elif rewritten.startswith('you '):
                rewritten = "I appreciate that you " + rewritten[4:]
        
        # Make friendly
        if target_style == "friendly":
            # Add friendly connectors
            if not rewritten.endswith(('!', '?')):
                friendly_endings = [
                    "! I hope this helps!",
                    "! Let me know if you need anything else!",
                    "! Thanks so much!",
                    "! Have a great day!"
                ]
                if not any(ending in rewritten.lower() for ending in friendly_endings):
                    rewritten += "! " + np.random.choice(friendly_endings)
        
        # Ensure proper capitalization
        rewritten = rewritten.strip()
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:]
        
        return rewritten
    
    def _generate_with_policy(self, text: str, max_length: int = 100) -> str:
        """
        Generate rewritten text using the policy network
        
        Args:
            text: Input text
            max_length: Maximum length of generated text
            
        Returns:
            Generated rewritten text
        """
        if not self.tokenizer or not self.policy_network:
            return self._apply_rule_based_rewriting(text)
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        # Get policy network output
        with torch.no_grad():
            logits, value, hidden_states = self.policy_network(
                inputs['input_ids'], 
                inputs['attention_mask']
            )
        
        # Generate using policy network
        generated_ids = []
        current_ids = inputs['input_ids']
        
        for _ in range(max_length):
            with torch.no_grad():
                logits, _, _ = self.policy_network(current_ids)
            
            # Get next token probabilities
            next_token_logits = logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(next_token_probs, 1)
            
            # Check for end token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Truncate if too long
            if current_ids.size(1) > 512:
                current_ids = current_ids[:, -512:]
        
        # Decode generated text
        if generated_ids:
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()
        else:
            return self._apply_rule_based_rewriting(text)
    
    def rewrite_text(self, text: str, target_style: str = "polite", use_policy: bool = False) -> str:
        """
        Rewrite text to be more polite, positive, or friendly
        
        Args:
            text: Input text to rewrite
            target_style: Target style ("polite", "positive", "friendly")
            use_policy: Whether to use policy network or rule-based approach
            
        Returns:
            Rewritten text
        """
        if not text or not text.strip():
            return text
        
        # Clean input text
        text = text.strip()
        
        # Use rule-based approach by default (preserves meaning and intent)
        rewritten = self._apply_rule_based_rewriting(text, target_style)
        
        # Only use policy network if specifically requested and trained
        if use_policy and self.policy_network and self.tokenizer:
            try:
                policy_rewritten = self._generate_with_policy(text)
                if policy_rewritten and len(policy_rewritten) > 0 and self._is_meaningful_text(policy_rewritten):
                    return policy_rewritten
            except Exception as e:
                print(f"Policy network generation failed: {e}")
        
        return rewritten
    
    def _is_meaningful_text(self, text: str) -> bool:
        """Check if the generated text is meaningful (not random tokens)"""
        if not text or len(text) < 3:
            return False
        
        # Check for common English words
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they']
        
        words = text.lower().split()
        if len(words) < 2:
            return False
        
        # Count meaningful words
        meaningful_count = sum(1 for word in words if word in common_words or len(word) > 2)
        
        # If less than 30% of words are meaningful, consider it random
        return meaningful_count / len(words) > 0.3
    
    def get_rewriting_suggestions(self, text: str) -> Dict[str, str]:
        """
        Get multiple rewriting suggestions
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with different style suggestions
        """
        return {
            'polite': self.rewrite_text(text, 'polite'),
            'positive': self.rewrite_text(text, 'positive'),
            'friendly': self.rewrite_text(text, 'friendly')
        }
    
    def update_policy(self, rewards: List[float], states: List[torch.Tensor], actions: List[torch.Tensor]):
        """
        Update policy network based on rewards (to be called by PPO)
        
        Args:
            rewards: List of rewards for each action
            states: List of state tensors
            actions: List of action tensors
        """
        # This will be implemented in the PPO module
        pass
    
    def save_model(self, path: str):
        """Save the policy network"""
        if self.policy_network:
            torch.save({
                'policy_state_dict': self.policy_network.state_dict(),
                'vocab_size': self.vocab_size,
                'model_name': self.model_name
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the policy network"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
