"""
PPO (Proximal Policy Optimization) Trainer for RLHF
Implements PPO algorithm to train the text rewriting policy network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
from collections import deque
import gc

class PPOTrainer:
    """
    PPO trainer for reinforcement learning from human feedback
    """
    def __init__(
        self,
        policy_network: nn.Module,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize PPO trainer
        
        Args:
            policy_network: The policy network to train
            learning_rate: Learning rate for optimizer
            clip_ratio: PPO clipping ratio
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on
        """
        self.policy_network = policy_network
        self.device = self._get_device(device)
        
        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=1000)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def add_experience(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ):
        """
        Add experience to the buffer
        
        Args:
            states: State tensors
            actions: Action tensors
            rewards: Reward tensors
            old_log_probs: Old log probabilities
            values: Value estimates
            dones: Done flags
        """
        experience = {
            'states': states.detach(),
            'actions': actions.detach(),
            'rewards': rewards.detach(),
            'old_log_probs': old_log_probs.detach(),
            'values': values.detach(),
            'dones': dones.detach()
        }
        self.experience_buffer.append(experience)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            dones: Done flags
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PPO policy loss
        
        Args:
            states: State tensors
            actions: Action tensors
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            
        Returns:
            Tuple of (policy_loss, kl_divergence, clip_fraction)
        """
        # Get current policy output
        logits, _, _ = self.policy_network(states)
        
        # Compute current log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        current_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute probability ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Compute clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute KL divergence
        kl_divergence = (old_log_probs - current_log_probs).mean()
        
        # Compute clip fraction
        clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
        
        return policy_loss, kl_divergence, clip_fraction
    
    def compute_value_loss(
        self,
        states: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss
        
        Args:
            states: State tensors
            returns: Target returns
            
        Returns:
            Value loss
        """
        _, values, _ = self.policy_network(states)
        value_loss = F.mse_loss(values.squeeze(), returns)
        return value_loss
    
    def compute_entropy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy bonus
        
        Args:
            states: State tensors
            
        Returns:
            Entropy loss (negative entropy)
        """
        logits, _, _ = self.policy_network(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -entropy  # Negative because we want to maximize entropy
    
    def update_policy(self, batch_size: int = 32, num_epochs: int = 4) -> Dict[str, float]:
        """
        Update the policy network using PPO
        
        Args:
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary of training statistics
        """
        if len(self.experience_buffer) < batch_size:
            return {}
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Concatenate batch data
        states = torch.cat([exp['states'] for exp in batch])
        actions = torch.cat([exp['actions'] for exp in batch])
        rewards = torch.cat([exp['rewards'] for exp in batch])
        old_log_probs = torch.cat([exp['old_log_probs'] for exp in batch])
        values = torch.cat([exp['values'] for exp in batch])
        dones = torch.cat([exp['dones'] for exp in batch])
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Training loop
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            states_shuffled = states[indices]
            actions_shuffled = actions[indices]
            old_log_probs_shuffled = old_log_probs[indices]
            advantages_shuffled = advantages[indices]
            returns_shuffled = returns[indices]
            
            # Compute losses
            policy_loss, kl_divergence, clip_fraction = self.compute_policy_loss(
                states_shuffled, actions_shuffled, old_log_probs_shuffled, advantages_shuffled
            )
            value_loss = self.compute_value_loss(states_shuffled, returns_shuffled)
            entropy_loss = self.compute_entropy_loss(states_shuffled)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Store statistics
            epoch_stats['policy_loss'].append(policy_loss.item())
            epoch_stats['value_loss'].append(value_loss.item())
            epoch_stats['entropy_loss'].append(entropy_loss.item())
            epoch_stats['total_loss'].append(total_loss.item())
            epoch_stats['kl_divergence'].append(kl_divergence.item())
            epoch_stats['clip_fraction'].append(clip_fraction.item())
        
        # Average statistics
        avg_stats = {key: np.mean(values) for key, values in epoch_stats.items()}
        
        # Update global statistics
        for key, value in avg_stats.items():
            self.training_stats[key].append(value)
        
        # Clear experience buffer to save memory
        self.experience_buffer.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return avg_stats
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.training_stats.copy()
    
    def reset_stats(self):
        """Reset training statistics"""
        for key in self.training_stats:
            self.training_stats[key].clear()
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint['training_stats']
            print(f"Checkpoint loaded from {path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
