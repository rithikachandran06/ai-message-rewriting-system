"""
Sentiment Analysis Module for Text Analysis
Analyzes sentiment, tone, intent, and meaning of input text
"""

import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

class SentimentAnalyzer:
    def __init__(self, device: str = "auto"):
        """
        Initialize the sentiment analyzer with multiple models
        
        Args:
            device: Device to run models on ("auto", "cpu", "cuda")
        """
        self.device = self._get_device(device)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load lightweight models for 8GB RAM
        try:
            # Sentiment analysis model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                max_length=512,
                truncation=True
            )
        except Exception as e:
            print(f"Warning: Could not load sentiment model: {e}")
            self.sentiment_pipeline = None
        
        # Intent classification (using a lightweight model)
        try:
            self.intent_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/DialoGPT-medium",
                num_labels=5  # 5 intent categories
            ).to(self.device)
        except Exception as e:
            print(f"Warning: Could not load intent model: {e}")
            self.intent_model = None
            self.intent_tokenizer = None
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of the text
        
        Returns:
            Tuple of (sentiment_score, sentiment_label)
        """
        if not text.strip():
            return 0.0, "neutral"
        
        # Use VADER for sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        compound_score = vader_scores['compound']
        
        # Use TextBlob for additional sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Combine scores
        combined_score = (compound_score + polarity) / 2
        
        # Determine label
        if combined_score >= 0.1:
            label = "positive"
        elif combined_score <= -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return combined_score, label
    
    def analyze_tone(self, text: str) -> Tuple[float, str]:
        """
        Analyze tone of the text
        
        Returns:
            Tuple of (tone_score, tone_label)
        """
        if not text.strip():
            return 0.0, "neutral"
        
        # Simple tone analysis based on linguistic features
        text_lower = text.lower()
        
        # Positive tone indicators
        positive_words = ['please', 'thank', 'appreciate', 'help', 'support', 'great', 'excellent', 'wonderful']
        negative_words = ['hate', 'terrible', 'awful', 'horrible', 'disgusting', 'angry', 'frustrated']
        polite_words = ['please', 'thank you', 'would you', 'could you', 'may i', 'excuse me']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        polite_count = sum(1 for phrase in polite_words if phrase in text_lower)
        
        # Calculate tone score
        tone_score = (positive_count - negative_count + polite_count * 0.5) / max(len(text.split()), 1)
        
        # Normalize to [-1, 1]
        tone_score = max(-1, min(1, tone_score * 10))
        
        # Determine tone label
        if tone_score >= 0.3:
            tone_label = "positive"
        elif tone_score <= -0.3:
            tone_label = "negative"
        else:
            tone_label = "neutral"
        
        return tone_score, tone_label
    
    def analyze_intent(self, text: str) -> Tuple[float, str]:
        """
        Analyze intent of the text
        
        Returns:
            Tuple of (intent_score, intent_label)
        """
        if not text.strip():
            return 0.0, "informational"
        
        text_lower = text.lower()
        
        # Intent categories
        intent_patterns = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', '?'],
            'request': ['can you', 'could you', 'please', 'help me', 'i need', 'i want'],
            'complaint': ['problem', 'issue', 'wrong', 'error', 'bug', 'not working'],
            'compliment': ['great', 'excellent', 'wonderful', 'amazing', 'love', 'perfect'],
            'informational': ['i think', 'i believe', 'according to', 'the fact is']
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            intent_scores[intent] = score
        
        # Find the intent with highest score
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # Normalize score
        intent_score = min(1.0, max_score / 3.0)
        
        return intent_score, best_intent
    
    def analyze_meaning(self, text: str) -> float:
        """
        Analyze meaning clarity and coherence
        
        Returns:
            Meaning score (0-1)
        """
        if not text.strip():
            return 0.0
        
        # Basic meaning analysis
        words = text.split()
        if len(words) < 2:
            return 0.3
        
        # Check for complete sentences
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        word_count = len(words)
        
        # Check for proper sentence structure
        has_capital = text[0].isupper() if text else False
        has_punctuation = any(p in text for p in '.!?')
        
        # Calculate meaning score
        meaning_score = 0.0
        
        # Sentence completeness
        if sentence_count > 0:
            meaning_score += 0.3
        
        # Proper capitalization
        if has_capital:
            meaning_score += 0.2
        
        # Proper punctuation
        if has_punctuation:
            meaning_score += 0.2
        
        # Word count (not too short, not too long)
        if 3 <= word_count <= 50:
            meaning_score += 0.3
        elif word_count > 50:
            meaning_score += 0.2  # Slightly penalize very long texts
        
        return min(1.0, meaning_score)
    
    def analyze_text(self, text: str) -> Dict:
        """
        Complete text analysis
        
        Returns:
            Dictionary with all analysis results
        """
        if not text or not text.strip():
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'tone_score': 0.0,
                'tone_label': 'neutral',
                'intent_score': 0.0,
                'intent_label': 'informational',
                'meaning_score': 0.0,
                'overall_score': 0.0
            }
        
        # Perform all analyses
        sentiment_score, sentiment_label = self.analyze_sentiment(text)
        tone_score, tone_label = self.analyze_tone(text)
        intent_score, intent_label = self.analyze_intent(text)
        meaning_score = self.analyze_meaning(text)
        
        # Calculate overall score (weighted average)
        overall_score = (
            abs(sentiment_score) * 0.3 +
            abs(tone_score) * 0.3 +
            intent_score * 0.2 +
            meaning_score * 0.2
        )
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'tone_score': tone_score,
            'tone_label': tone_label,
            'intent_score': intent_score,
            'intent_label': intent_label,
            'meaning_score': meaning_score,
            'overall_score': overall_score
        }
    
    def get_analysis_summary(self, text: str) -> str:
        """
        Get a human-readable summary of the analysis
        
        Returns:
            String summary of the analysis
        """
        analysis = self.analyze_text(text)
        
        summary = f"""
Text Analysis Summary:
- Sentiment: {analysis['sentiment_label']} (score: {analysis['sentiment_score']:.2f})
- Tone: {analysis['tone_label']} (score: {analysis['tone_score']:.2f})
- Intent: {analysis['intent_label']} (score: {analysis['intent_score']:.2f})
- Meaning Clarity: {analysis['meaning_score']:.2f}
- Overall Score: {analysis['overall_score']:.2f}
        """.strip()
        
        return summary
