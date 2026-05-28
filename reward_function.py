"""
Reward Function Module for RLHF
Computes rewards based on user ratings and text quality metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from sentiment_analyzer import SentimentAnalyzer

class RewardFunction:
    """
    Reward function that combines user ratings with automated quality metrics
    """
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        """
        Initialize reward function
        
        Args:
            sentiment_analyzer: Sentiment analyzer instance
        """
        self.sentiment_analyzer = sentiment_analyzer
        
        # Reward weights
        self.weights = {
            'user_rating': 0.4,      # User rating weight
            'sentiment_improvement': 0.2,  # Sentiment improvement weight
            'tone_improvement': 0.15,      # Tone improvement weight
            'meaning_preservation': 0.15,  # Meaning preservation weight
            'intent_preservation': 0.1     # Intent preservation weight
        }
        
        # Quality thresholds
        self.thresholds = {
            'min_sentiment_score': 0.3,
            'min_tone_score': 0.3,
            'min_meaning_score': 0.7,
            'min_intent_score': 0.5
        }
    
    def get_user_rating(self, original_text: str, rewritten_text: str) -> float:
        """
        Get user rating for the rewritten text
        
        Args:
            original_text: Original input text
            rewritten_text: Rewritten text
            
        Returns:
            User rating (0-1 scale)
        """
        print(f"\nOriginal text: {original_text}")
        print(f"Rewritten text: {rewritten_text}")
        print("\nPlease rate the rewritten text:")
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
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                return 0.5  # Default neutral rating
            except Exception as e:
                print(f"Error getting rating: {e}")
                return 0.5
    
    def compute_sentiment_improvement(self, original_text: str, rewritten_text: str) -> float:
        """
        Compute sentiment improvement reward
        
        Args:
            original_text: Original text
            rewritten_text: Rewritten text
            
        Returns:
            Sentiment improvement score (0-1)
        """
        try:
            original_analysis = self.sentiment_analyzer.analyze_text(original_text)
            rewritten_analysis = self.sentiment_analyzer.analyze_text(rewritten_text)
            
            # Calculate improvement
            sentiment_improvement = rewritten_analysis['sentiment_score'] - original_analysis['sentiment_score']
            
            # Normalize to [0, 1] range
            # Positive improvement gets higher reward
            normalized_improvement = (sentiment_improvement + 1) / 2
            
            return max(0, min(1, normalized_improvement))
        except Exception as e:
            print(f"Error computing sentiment improvement: {e}")
            return 0.5
    
    def compute_tone_improvement(self, original_text: str, rewritten_text: str) -> float:
        """
        Compute tone improvement reward
        
        Args:
            original_text: Original text
            rewritten_text: Rewritten text
            
        Returns:
            Tone improvement score (0-1)
        """
        try:
            original_analysis = self.sentiment_analyzer.analyze_text(original_text)
            rewritten_analysis = self.sentiment_analyzer.analyze_text(rewritten_text)
            
            # Calculate improvement
            tone_improvement = rewritten_analysis['tone_score'] - original_analysis['tone_score']
            
            # Normalize to [0, 1] range
            normalized_improvement = (tone_improvement + 1) / 2
            
            return max(0, min(1, normalized_improvement))
        except Exception as e:
            print(f"Error computing tone improvement: {e}")
            return 0.5
    
    def compute_meaning_preservation(self, original_text: str, rewritten_text: str) -> float:
        """
        Compute meaning preservation reward
        
        Args:
            original_text: Original text
            rewritten_text: Rewritten text
            
        Returns:
            Meaning preservation score (0-1)
        """
        try:
            # Simple word overlap analysis
            original_words = set(original_text.lower().split())
            rewritten_words = set(rewritten_text.lower().split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            original_words = original_words - stop_words
            rewritten_words = rewritten_words - stop_words
            
            if not original_words:
                return 0.5  # Neutral if no meaningful words
            
            # Calculate word overlap
            overlap = len(original_words.intersection(rewritten_words))
            total_original = len(original_words)
            
            # Word overlap score
            word_overlap_score = overlap / total_original if total_original > 0 else 0
            
            # Length similarity (shouldn't be too different)
            length_ratio = min(len(rewritten_text), len(original_text)) / max(len(rewritten_text), len(original_text))
            
            # Combine scores
            meaning_score = (word_overlap_score * 0.7 + length_ratio * 0.3)
            
            return max(0, min(1, meaning_score))
        except Exception as e:
            print(f"Error computing meaning preservation: {e}")
            return 0.5
    
    def compute_intent_preservation(self, original_text: str, rewritten_text: str) -> float:
        """
        Compute intent preservation reward
        
        Args:
            original_text: Original text
            rewritten_text: Rewritten text
            
        Returns:
            Intent preservation score (0-1)
        """
        try:
            original_analysis = self.sentiment_analyzer.analyze_text(original_text)
            rewritten_analysis = self.sentiment_analyzer.analyze_text(rewritten_text)
            
            # Check if intent labels match
            intent_match = 1.0 if original_analysis['intent_label'] == rewritten_analysis['intent_label'] else 0.0
            
            # Check intent score similarity
            intent_score_diff = abs(original_analysis['intent_score'] - rewritten_analysis['intent_score'])
            intent_score_similarity = max(0, 1 - intent_score_diff)
            
            # Combine scores
            intent_preservation = (intent_match * 0.6 + intent_score_similarity * 0.4)
            
            return intent_preservation
        except Exception as e:
            print(f"Error computing intent preservation: {e}")
            return 0.5
    
    def compute_quality_bonus(self, rewritten_text: str) -> float:
        """
        Compute quality bonus based on text characteristics
        
        Args:
            rewritten_text: Rewritten text
            
        Returns:
            Quality bonus score (0-1)
        """
        try:
            analysis = self.sentiment_analyzer.analyze_text(rewritten_text)
            
            # Check if text meets quality thresholds
            quality_score = 0.0
            
            if analysis['sentiment_score'] >= self.thresholds['min_sentiment_score']:
                quality_score += 0.25
            
            if analysis['tone_score'] >= self.thresholds['min_tone_score']:
                quality_score += 0.25
            
            if analysis['meaning_score'] >= self.thresholds['min_meaning_score']:
                quality_score += 0.25
            
            if analysis['intent_score'] >= self.thresholds['min_intent_score']:
                quality_score += 0.25
            
            return quality_score
        except Exception as e:
            print(f"Error computing quality bonus: {e}")
            return 0.0
    
    def compute_reward(
        self,
        original_text: str,
        rewritten_text: str,
        user_rating: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward for the rewritten text
        
        Args:
            original_text: Original input text
            rewritten_text: Rewritten text
            user_rating: Optional user rating (if None, will prompt user)
            
        Returns:
            Tuple of (total_reward, component_scores)
        """
        # Get user rating if not provided
        if user_rating is None:
            user_rating = self.get_user_rating(original_text, rewritten_text)
        
        # Compute component scores
        sentiment_improvement = self.compute_sentiment_improvement(original_text, rewritten_text)
        tone_improvement = self.compute_tone_improvement(original_text, rewritten_text)
        meaning_preservation = self.compute_meaning_preservation(original_text, rewritten_text)
        intent_preservation = self.compute_intent_preservation(original_text, rewritten_text)
        quality_bonus = self.compute_quality_bonus(rewritten_text)
        
        # Compute weighted reward
        total_reward = (
            self.weights['user_rating'] * user_rating +
            self.weights['sentiment_improvement'] * sentiment_improvement +
            self.weights['tone_improvement'] * tone_improvement +
            self.weights['meaning_preservation'] * meaning_preservation +
            self.weights['intent_preservation'] * intent_preservation +
            quality_bonus * 0.1  # Small quality bonus
        )
        
        # Component scores for analysis
        component_scores = {
            'user_rating': user_rating,
            'sentiment_improvement': sentiment_improvement,
            'tone_improvement': tone_improvement,
            'meaning_preservation': meaning_preservation,
            'intent_preservation': intent_preservation,
            'quality_bonus': quality_bonus,
            'total_reward': total_reward
        }
        
        return total_reward, component_scores
    
    def should_continue_training(self, recent_rewards: List[float], threshold: float = 0.8) -> bool:
        """
        Determine if training should continue based on recent rewards
        
        Args:
            recent_rewards: List of recent reward values
            threshold: Reward threshold for stopping training
            
        Returns:
            True if training should continue, False otherwise
        """
        if len(recent_rewards) < 5:
            return True
        
        # Check if recent rewards are consistently high
        recent_avg = np.mean(recent_rewards[-5:])
        return recent_avg < threshold
    
    def get_reward_summary(self, component_scores: Dict[str, float]) -> str:
        """
        Get a human-readable summary of reward components
        
        Args:
            component_scores: Dictionary of component scores
            
        Returns:
            String summary
        """
        summary = f"""
Reward Analysis:
- User Rating: {component_scores['user_rating']:.2f}
- Sentiment Improvement: {component_scores['sentiment_improvement']:.2f}
- Tone Improvement: {component_scores['tone_improvement']:.2f}
- Meaning Preservation: {component_scores['meaning_preservation']:.2f}
- Intent Preservation: {component_scores['intent_preservation']:.2f}
- Quality Bonus: {component_scores['quality_bonus']:.2f}
- Total Reward: {component_scores['total_reward']:.2f}
        """.strip()
        
        return summary
