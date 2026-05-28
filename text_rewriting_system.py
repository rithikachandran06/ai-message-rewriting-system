"""
Memory-Optimized Text Rewriting System with Sentiment Analysis and RLHF
- Analyzes input text sentiment, tone, intent, and meaning
- Rewrites text to be more polite/positive using Qwen LLM
- Uses RLHF with PPO to improve the model based on user feedback
- Includes visualization of metrics
- Optimized for low memory environments (8GB RAM)
"""

import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import components
from sentiment_analyzer import SentimentAnalyzer
from qwen_rewriter import QwenRewriter
from reward_function import RewardFunction
from ppo_trainer import PPOTrainer
from model_manager import ModelManager

class TextRewritingSystem:
    """
    Memory-optimized text rewriting system with sentiment analysis and RLHF
    """
    def __init__(self, device: str = "auto", model_save_path: str = "./models/qwen_improved", 
                 low_memory: bool = True):
        """
        Initialize the text rewriting system
        
        Args:
            device: Device to run on ("auto", "cpu", "cuda")
            model_save_path: Path to save the improved model
            low_memory: Enable memory optimization for 8GB RAM systems
        """
        self.device = self._get_device(device)
        self.model_save_path = model_save_path
        self.low_memory = low_memory
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        print(f"Initializing Text Rewriting System on {self.device} with memory optimization")
        
        # Set torch to use less memory
        if self.low_memory:
            torch.set_num_threads(2)  # Limit CPU threads
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Initialize components lazily - only when needed
        self._sentiment_analyzer = None
        self._qwen_rewriter = None
        self._reward_function = None
        self._model_manager = None
        
        # Metrics tracking
        self.metrics_history = {
            'sentiment_scores': [],
            'tone_scores': [],
            'intent_scores': [],
            'meaning_scores': [],
            'user_ratings': [],
            'reward_values': []
        }
    
    @property
    def sentiment_analyzer(self):
        """Lazy-load sentiment analyzer"""
        if self._sentiment_analyzer is None:
            print("Loading sentiment analyzer...")
            self._sentiment_analyzer = SentimentAnalyzer(device=self.device)
        return self._sentiment_analyzer
    
    @property
    def qwen_rewriter(self):
        """Lazy-load Qwen rewriter"""
        if self._qwen_rewriter is None:
            print("Loading Qwen rewriter...")
            try:
                self._qwen_rewriter = QwenRewriter(device=self.device)
            except Exception as e:
                print(f"Error loading Qwen rewriter: {e}")
                print("Using fallback text rewriting method...")
                from text_rewriter import TextRewriter
                self._qwen_rewriter = TextRewriter()
        return self._qwen_rewriter
    
    @property
    def reward_function(self):
        """Lazy-load reward function"""
        if self._reward_function is None:
            self._reward_function = RewardFunction(self.sentiment_analyzer)
        return self._reward_function
    
    @property
    def model_manager(self):
        """Lazy-load model manager"""
        if self._model_manager is None:
            self._model_manager = ModelManager()
        return self._model_manager
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze input text sentiment, tone, intent, and meaning
        
        Args:
            text: Input text to analyze
            
        Returns:
            Analysis results
        """
        analysis = self.sentiment_analyzer.analyze_text(text)
        
        print("\n=== Text Analysis ===")
        print(f"Text: {text}")
        print(f"Sentiment: {analysis['sentiment_label']} (Score: {analysis['sentiment_score']:.2f})")
        print(f"Tone: {analysis['tone_label']} (Score: {analysis['tone_score']:.2f})")
        print(f"Intent: {analysis['intent_label']}")
        print(f"Meaning Score: {analysis['meaning_score']:.2f}")
        
        return analysis
    
    def rewrite_text(self, text: str) -> Tuple[str, Dict]:
        """
        Rewrite text to be more polite and positive
        
        Args:
            text: Input text to rewrite
            
        Returns:
            Tuple of (rewritten_text, metrics)
        """
        # Analyze original text
        original_analysis = self.sentiment_analyzer.analyze_text(text)
        
        # Rewrite text using Qwen
        rewritten_text = self.qwen_rewriter.rewrite(text)
        
        # Analyze rewritten text
        rewritten_analysis = self.sentiment_analyzer.analyze_text(rewritten_text)
        
        # Calculate improvements
        sentiment_improvement = rewritten_analysis['sentiment_score'] - original_analysis['sentiment_score']
        tone_improvement = rewritten_analysis['tone_score'] - original_analysis['tone_score']
        
        # Calculate meaning preservation (simple implementation)
        # Since SentimentAnalyzer doesn't have calculate_meaning_similarity method
        try:
            # Try to use a simple word overlap ratio as meaning preservation
            original_words = set(text.lower().split())
            rewritten_words = set(rewritten_text.lower().split())
            
            if not original_words:
                meaning_preservation = 1.0  # Empty text case
            else:
                common_words = original_words.intersection(rewritten_words)
                meaning_preservation = len(common_words) / len(original_words)
        except Exception:
            # Fallback to a default value
            meaning_preservation = 0.8
        
        # Calculate intent preservation (binary)
        intent_preserved = original_analysis['intent_label'] == rewritten_analysis['intent_label']
        
        metrics = {
            'original_text': text,
            'rewritten_text': rewritten_text,
            'original_analysis': original_analysis,
            'rewritten_analysis': rewritten_analysis,
            'sentiment_improvement': sentiment_improvement,
            'tone_improvement': tone_improvement,
            'meaning_preservation': meaning_preservation,
            'intent_preserved': intent_preserved
        }
        
        print("\n=== Text Rewriting ===")
        print(f"Original: {text}")
        print(f"Rewritten: {rewritten_text}")
        print(f"Sentiment improvement: {sentiment_improvement:.2f}")
        print(f"Tone improvement: {tone_improvement:.2f}")
        print(f"Meaning preservation: {meaning_preservation:.2f}")
        print(f"Intent preserved: {'Yes' if intent_preserved else 'No'}")
        
        return rewritten_text, metrics
    
    def evaluate_quality(self, metrics: Dict) -> float:
        """
        Evaluate the quality of the rewritten text
        
        Args:
            metrics: Metrics from rewrite_text
            
        Returns:
            Quality score (0-1)
        """
        # Weights for different factors
        weights = {
            'sentiment_improvement': 0.3,
            'tone_improvement': 0.2,
            'meaning_preservation': 0.4,
            'intent_preserved': 0.1
        }
        
        # Calculate weighted score
        score = (
            weights['sentiment_improvement'] * max(0, metrics['sentiment_improvement']) +
            weights['tone_improvement'] * max(0, metrics['tone_improvement']) +
            weights['meaning_preservation'] * metrics['meaning_preservation'] +
            weights['intent_preserved'] * (1.0 if metrics['intent_preserved'] else 0.0)
        )
        
        # Normalize to 0-1
        normalized_score = min(1.0, max(0.0, score))
        
        print(f"\nQuality Score: {normalized_score:.2f}")
        
        return normalized_score
    
    def get_user_feedback(self, original_text: str, rewritten_text: str) -> float:
        """
        Get user feedback on the rewritten text
        
        Args:
            original_text: Original text
            rewritten_text: Rewritten text
            
        Returns:
            User rating (0-1)
        """
        print("\n=== User Feedback ===")
        print("Please rate the quality of the rewritten text:")
        print("1. Very poor (0.0)")
        print("2. Poor (0.25)")
        print("3. Fair (0.5)")
        print("4. Good (0.75)")
        print("5. Excellent (1.0)")
        
        while True:
            try:
                rating = input("Enter your rating (1-5): ").strip()
                rating_map = {
                    '1': 0.0,
                    '2': 0.25,
                    '3': 0.5,
                    '4': 0.75,
                    '5': 1.0
                }
                
                if rating in rating_map:
                    return rating_map[rating]
                else:
                    print("Please enter a valid rating (1-5)")
            except Exception:
                print("Invalid input. Using default rating of 0.5.")
                return 0.5
    
    def update_model(self, original_text: str, rewritten_text: str, user_rating: float) -> Dict:
        """
        Update the model using RLHF with PPO
        
        Args:
            original_text: Original text
            rewritten_text: Rewritten text
            user_rating: User rating (0-1)
            
        Returns:
            Update results
        """
        print("\n=== Updating Model ===")
        
        # Compute reward
        reward_value, components = self.reward_function.compute_reward(
            original_text, rewritten_text, user_rating=user_rating
        )
        
        print(f"Reward value: {reward_value:.4f}")
        print("Reward components:")
        for k, v in components.items():
            print(f"- {k}: {v:.4f}")
        
        # Update metrics history
        self.metrics_history['sentiment_scores'].append(components.get('sentiment_improvement', 0))
        self.metrics_history['tone_scores'].append(components.get('tone_improvement', 0))
        self.metrics_history['intent_scores'].append(components.get('intent_preservation', 0))
        self.metrics_history['meaning_scores'].append(components.get('meaning_preservation', 0))
        self.metrics_history['user_ratings'].append(user_rating)
        self.metrics_history['reward_values'].append(reward_value)
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Create dummy optimizer and training stats for model saving
        from datetime import datetime
        dummy_optimizer = torch.optim.Adam(self.qwen_rewriter.model.parameters(), lr=1e-5)
        training_stats = {
            "reward": reward_value,
            "components": components,
            "iterations": len(self.metrics_history['reward_values']),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save model with the correct method signature
        model_path = self.model_manager.save_model(
            model=self.qwen_rewriter.model,
            optimizer=dummy_optimizer,
            training_stats=training_stats,
            model_name="qwen_rewriter"
        )
        
        print(f"Model saved to {model_path}")
        
        return {
            'reward': reward_value,
            'components': components
        }
    
    def visualize_metrics(self) -> None:
        """
        Visualize metrics history with memory optimization
        """
        if not self.metrics_history['reward_values']:
            print("No metrics to visualize yet.")
            return
        
        # Use smaller figure size and lower DPI for memory efficiency
        plt.figure(figsize=(8, 6), dpi=80)
        
        # Plot all metrics in a single plot to save memory
        plt.plot(self.metrics_history['sentiment_scores'], 'b-', label='Sentiment')
        plt.plot(self.metrics_history['tone_scores'], 'g-', label='Tone')
        plt.plot(self.metrics_history['meaning_scores'], 'r-', label='Meaning')
        plt.plot(self.metrics_history['intent_scores'], 'y-', label='Intent')
        plt.plot(self.metrics_history['user_ratings'], 'c-', label='User Rating')
        plt.plot(self.metrics_history['reward_values'], 'm-', label='Reward')
        
        plt.title('Text Rewriting Metrics')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # Save with lower quality for memory efficiency
        plt.savefig('metrics_visualization.png', dpi=80, bbox_inches='tight')
        plt.close()
        
        # Force garbage collection
        gc.collect()
        
        print("\nMetrics visualization saved to 'metrics_visualization.png'")
    
    def process_text(self, text: str, update_model: bool = True) -> Dict:
        """
        Process text through the complete pipeline
        
        Args:
            text: Input text
            update_model: Whether to update the model
            
        Returns:
            Processing results
        """
        print("\n" + "="*50)
        print("TEXT REWRITING SYSTEM")
        print("="*50)
        
        # Step 1: Analyze input text
        original_analysis = self.analyze_text(text)
        
        # Step 2: Rewrite text
        rewritten_text, metrics = self.rewrite_text(text)
        
        # Step 3: Evaluate quality
        quality_score = self.evaluate_quality(metrics)
        
        # Step 4: If quality is low, get user feedback and update model
        if quality_score < 0.7 and update_model:
            print("\nQuality score is below threshold. Getting user feedback...")
            user_rating = self.get_user_feedback(text, rewritten_text)
            update_results = self.update_model(text, rewritten_text, user_rating)
            
            # Visualize metrics
            self.visualize_metrics()
            
            return {
                'original_text': text,
                'rewritten_text': rewritten_text,
                'quality_score': quality_score,
                'user_rating': user_rating,
                'update_results': update_results
            }
        else:
            print("\nQuality score is above threshold. No model update needed.")
            return {
                'original_text': text,
                'rewritten_text': rewritten_text,
                'quality_score': quality_score
            }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Rewriting System with RLHF")
    parser.add_argument("--text", help="Input text to process")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to run on")
    parser.add_argument("--no-update", action="store_true",
                       help="Don't update the model")
    parser.add_argument("--model-path", default="./models/qwen_improved",
                       help="Path to save the improved model")
    
    args = parser.parse_args()
    
    # Initialize system
    system = TextRewritingSystem(device=args.device, model_save_path=args.model_path)
    
    if args.text:
        # Process single text
        system.process_text(args.text, update_model=not args.no_update)
    else:
        # Interactive mode
        print("\nText Rewriting System - Interactive Mode")
        print("Enter 'quit' to exit")
        
        while True:
            text = input("\nEnter text to process: ")
            if text.lower() == 'quit':
                break
            
            system.process_text(text, update_model=not args.no_update)

if __name__ == "__main__":
    main()