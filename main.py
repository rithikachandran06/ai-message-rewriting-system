"""
Main Application for Text Rewriting with RLHF
Complete system for sentiment analysis, text rewriting, and RLHF training
"""

import sys
import os

# Windows-specific memory optimizations - MUST be before torch import
if sys.platform == "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # Set environment variables before torch is imported
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import torch
import traceback
import argparse
import gc
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

# Import utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from utils.logger import get_logger
    from utils.memory_utils import cleanup_memory, monitor_memory, MemoryContext
    from config import app_config, Constants
except ImportError:
    # Fallback if utils not available
    import logging
    def get_logger(name="text_rewriter"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def cleanup_memory(device="auto", verbose=True):
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def monitor_memory(device="auto", threshold_gb=7.0):
        return 0.0, False
    
    class MemoryContext:
        def __init__(self, device="auto", cleanup_on_exit=True):
            self.device = device
            self.cleanup_on_exit = cleanup_on_exit
        def __enter__(self): return self
        def __exit__(self, *args): return False
    
    class Constants:
        RATING_MAP = {'1': 0.0, '2': 0.25, '3': 0.5, '4': 0.75, '5': 1.0}
    
    app_config = type('Config', (), {'reward_threshold': 0.75, 'max_iterations': 5})()

# Windows-specific PyTorch thread settings - with error handling
if sys.platform == "win32":
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass  # Already set or parallel work started
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Already set or parallel work started

from sentiment_analyzer import SentimentAnalyzer
from qwen_rewriter import QwenRewriter
from main_training_system import TextRewritingTrainer
from model_manager import ModelManager

class TextRewritingApp:
    """
    Main application class for text rewriting with RLHF
    """
    def __init__(self, device: str = "auto"):
        """
        Initialize the application
        
        Args:
            device: Device to run on ("auto", "cpu", "cuda")
        """
        self.device = self._get_device(device)
        self.logger = get_logger("TextRewritingApp")
        self.logger.info(f"Initializing Text Rewriting App on {self.device}")
        
        # Initialize components
        try:
            # Load sentiment analyzer (lightweight)
            self.sentiment_analyzer = SentimentAnalyzer(device=self.device)
            self.model_manager = ModelManager()
            
            # Initialize text_rewriter and trainer as None by default
            # They can be preloaded or loaded on demand
            self.text_rewriter = None
            self.trainer = None
            
            self.logger.info("Components initialized")
            self.logger.info("Using Qwen LLM for text rewriting")
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize application: {e}") from e
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def analyze_text(self, text: str) -> dict:
        """
        Analyze input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Analysis results
        """
        try:
            if not text or not text.strip():
                return {"error": "Empty text provided"}
            
            analysis = self.sentiment_analyzer.analyze_text(text)
            summary = self.sentiment_analyzer.get_analysis_summary(text)
            
            return {
                "success": True,
                "analysis": analysis,
                "summary": summary
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def rewrite_text(self, text: str) -> dict:
        """
        Rewrite text to be more polite, positive, or friendly
        
        Args:
            text: Input text to rewrite
            target_style: Target style ("polite", "positive", "friendly")
            
        Returns:
            Rewriting results
        """
        try:
            if not text or not text.strip():
                return {"error": "Empty text provided"}
            
            # Lazy load text rewriter if needed
            if self.text_rewriter is None:
                print("Loading Qwen model...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    self.text_rewriter = QwenRewriter(device=self.device)
                except Exception as e:
                    print(f"Error loading Qwen model: {e}")
                    import traceback
                    traceback.print_exc()
                    gc.collect()
                    return {"success": False, "error": f"Failed to load Qwen model: {str(e)}"}
            
            # Analyze original text
            original_analysis = self.sentiment_analyzer.analyze_text(text)
            
            # Generate multiple candidates and select the best (no training)
            best_rewrite = None
            best_analysis = None
            best_meaning = 0.0
            best_score = -1e9
            
            try:
                candidates_result = self.generate_multiple_versions(text, num_versions=5)
                if candidates_result.get("success"):
                    versions = candidates_result.get("versions", [])
                else:
                    versions = []
            except Exception:
                versions = []
            
            # Fallback to single generation if multi fails
            if not versions:
                versions = [self.text_rewriter.rewrite(text)]

            # For violent inputs, add safe deterrent templates to candidate pool
            violent_keywords = {"kill", "hurt", "harm", "shoot", "stab", "destroy", "attack"}
            if any(w in text.lower() for w in violent_keywords):
                for t in self._generate_safe_deterrent_candidates(text):
                    if t not in versions:
                        versions.append(t)
            
            for candidate in versions:
                if not candidate or not candidate.strip():
                    continue
                # Enforce one-sentence output for fair comparison
                candidate = candidate.strip().split("\n\n")[0].split("\n")[0]
                if "." in candidate:
                    candidate = candidate.split(".", 1)[0].strip() + "."
                analysis = self.sentiment_analyzer.analyze_text(candidate)
                meaning = self._compute_meaning_preservation(text, candidate)
                
                # Intent preservation hard constraint/score
                original_intent = original_analysis.get('intent_label')
                candidate_intent = analysis.get('intent_label')
                intent_match = 1.0 if candidate_intent == original_intent else 0.0
                
                # Penalize contradiction via negation if original has none (with safe deterrent exceptions)
                negation_tokens = {"not", "never", "won't", "can't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't", "no", "n't"}
                orig_has_neg = any(tok in text.lower() for tok in negation_tokens)
                cand_has_neg = any(tok in candidate.lower() for tok in negation_tokens)
                negation_penalty = 0.25 if (not orig_has_neg and cand_has_neg) else 0.0
                
                # Encourage de-escalation for violent/threatening phrases
                violent_keywords = {"kill", "hurt", "harm", "shoot", "stab", "destroy", "attack"}
                orig_violence = sum(1 for w in violent_keywords if w in text.lower())
                cand_violence = sum(1 for w in violent_keywords if w in candidate.lower())
                deescalation_bonus = 0.2 if orig_violence > 0 and cand_violence == 0 else 0.0
                # Safe deterrent phrases are allowed and slightly preferred
                safe_deterrent_markers = (
                    "won't hesitate to take appropriate action",
                    "won't hesitate to deal with your actions",
                    "actions will have consequences",
                    "there will be consequences"
                )
                cand_lower = candidate.lower()
                if orig_violence > 0 and any(m in cand_lower for m in safe_deterrent_markers):
                    # Remove negation penalty for accepted deterrent phrasing
                    negation_penalty = 0.0
                    deescalation_bonus += 0.1
                # Prefer emotion/request framing
                deescalation_markers = ("i feel", "i'm upset", "i am upset", "i'm angry", "i am angry", "i'm frustrated", "i am frustrated")
                if orig_violence > 0 and any(m in candidate.lower() for m in deescalation_markers):
                    deescalation_bonus += 0.1

                # Penalize questions/offers/requests or added logistics that change intent
                add_penalty_phrases = [
                    "please let me know",
                    "let me know",
                    "if you need my help",
                    "help with something",
                    "urgent",
                    "immediate attention",
                    "can i",
                    "could i",
                    "shall i",
                    "may i",
                    "?"
                ]
                added_penalty = 0.0
                for p in add_penalty_phrases:
                    if p in candidate.lower():
                        added_penalty += 0.1
                # Score prioritizes intent, meaning, and safety; then tone/sentiment
                sentiment_impr = analysis['sentiment_score'] - original_analysis['sentiment_score']
                tone_impr = analysis['tone_score'] - original_analysis['tone_score']
                overall_impr = analysis['overall_score'] - original_analysis['overall_score']
                # Brevity preference: penalize big length increases
                len_orig = max(1, len(text))
                len_cand = len(candidate)
                length_ratio = len_cand / len_orig
                brevity_penalty = 0.0
                if length_ratio > 1.2:
                    brevity_penalty = min(0.3, (length_ratio - 1.2) * 0.5)

                score = 0.20 * overall_impr + 0.40 * meaning + 0.20 * intent_match + 0.20 * (sentiment_impr + tone_impr) / 2.0
                score += deescalation_bonus
                score -= negation_penalty
                score -= added_penalty
                score -= brevity_penalty
                # Prefer candidates that preserve meaning reasonably well
                if meaning < 0.6:
                    score -= 0.2
                # Strongly penalize intent changes
                if intent_match == 0.0:
                    score -= 0.4
                if score > best_score:
                    best_score = score
                    best_rewrite = candidate
                    best_analysis = analysis
                    best_meaning = meaning
            
            # If everything failed, return graceful error
            if best_rewrite is None:
                return {"success": False, "error": "Failed to generate rewritten text"}
            
            return {
                "success": True,
                "original_text": text,
                "rewritten_text": best_rewrite,
                "original_analysis": original_analysis,
                "rewritten_analysis": best_analysis,
                "meaning_preservation": best_meaning,
                "improvement": {
                    "sentiment": best_analysis['sentiment_score'] - original_analysis['sentiment_score'],
                    "tone": best_analysis['tone_score'] - original_analysis['tone_score'],
                    "overall": best_analysis['overall_score'] - original_analysis['overall_score'],
                    "meaning_preserved": best_meaning >= 0.6
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Rewriting failed: {str(e)}"
            }
    
    def generate_multiple_versions(self, text: str, num_versions: int = 5) -> dict:
        """
        Generate multiple rewritten versions of the text
        
        Args:
            text: Input text to rewrite
            num_versions: Number of versions to generate
            
        Returns:
            Dictionary with versions and analysis
        """
        try:
            if not text or not text.strip():
                return {"error": "Empty text provided"}
            
            # Lazy load text rewriter if needed
            if self.text_rewriter is None:
                print("Loading Qwen model...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    self.text_rewriter = QwenRewriter(device=self.device)
                except Exception as e:
                    print(f"Error loading Qwen model: {e}")
                    return {"success": False, "error": f"Failed to load Qwen model: {str(e)}"}
            
            # Analyze original text
            original_analysis = self.sentiment_analyzer.analyze_text(text)
            
            # Generate multiple versions
            versions = self.text_rewriter.generate_multiple_versions(text, num_versions=num_versions)
            
            if not versions or len(versions) == 0:
                return {"success": False, "error": "Failed to generate rewritten versions"}
            
            # Analyze each version
            version_analyses = []
            meaning_scores = []
            
            for version in versions:
                version_analysis = self.sentiment_analyzer.analyze_text(version)
                meaning_score = self._compute_meaning_preservation(text, version)
                version_analyses.append(version_analysis)
                meaning_scores.append(meaning_score)
            
            return {
                "success": True,
                "original_text": text,
                "original_analysis": original_analysis,
                "versions": versions,
                "version_analyses": version_analyses,
                "meaning_scores": meaning_scores
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate versions: {str(e)}"
            }
    
    def _compute_meaning_preservation(self, original_text: str, rewritten_text: str) -> float:
        """
        Compute meaning preservation score between original and rewritten text
        
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
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'}
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
            length_ratio = min(len(rewritten_text), len(original_text)) / max(len(rewritten_text), len(original_text)) if max(len(rewritten_text), len(original_text)) > 0 else 0
            
            # Intent preservation check
            original_intent = self.sentiment_analyzer.analyze_text(original_text).get('intent_label', '')
            rewritten_intent = self.sentiment_analyzer.analyze_text(rewritten_text).get('intent_label', '')
            intent_match = 1.0 if original_intent == rewritten_intent else 0.5
            
            # Combine scores
            meaning_score = (word_overlap_score * 0.5 + length_ratio * 0.2 + intent_match * 0.3)
            
            return max(0, min(1, meaning_score))
        except Exception as e:
            print(f"Error computing meaning preservation: {e}")
            return 0.5

    def _generate_safe_deterrent_candidates(self, original_text: str) -> list:
        """
        Generate safe, neutral deterrent one-liners that keep deterrent intent
        without adding logistics, offers, or questions.
        """
        base_candidates = [
            "I won't hesitate to deal with your actions.",
            "Your actions will have consequences.",
            "I won't hesitate to take appropriate action.",
            "There will be consequences for these actions."
        ]
        # Keep unique and single sentence
        uniq = []
        for c in base_candidates:
            c1 = c.strip().split("\n\n")[0].split("\n")[0]
            if "." in c1:
                c1 = c1.split(".", 1)[0].strip() + "."
            if c1 not in uniq:
                uniq.append(c1)
        return uniq
    
    def process_with_selection_and_feedback(self, text: str, reward_threshold: float = 0.75, max_iterations: int = 5) -> dict:
        """
        Complete workflow: Analyze → Generate 5 versions → User selects → Feedback → Update model
        
        Args:
            text: Input text to process
            
        Returns:
            Processing results
        """
        try:
            if not text or not text.strip():
                return {"error": "Empty text provided"}
            
            print("\n" + "="*60)
            print("TEXT REWRITING WITH ITERATIVE FEEDBACK")
            print("="*60)
            print(f"Target reward threshold: {reward_threshold:.2f}")
            print(f"Maximum iterations: {max_iterations}")
            
            # Step 1: Analyze sentiment
            print("\n[Step 1] Analyzing input text...")
            original_analysis = self.sentiment_analyzer.analyze_text(text)
            
            sentiment_score = original_analysis['sentiment_score']
            print(f"Sentiment Score: {sentiment_score:.2f} ({original_analysis['sentiment_label']})")
            
            # Check if negative - if not, inform user
            if sentiment_score >= 0.0:
                print(f"\n⚠️  Text is not negative (score: {sentiment_score:.2f}).")
                print("Would you like to proceed with rewriting anyway?")
                proceed = input("Proceed? (yes/no): ").strip().lower()
                if proceed not in ['yes', 'y']:
                    return {"success": False, "message": "User chose not to proceed with non-negative text"}
            
            # Lazy load text rewriter if needed
            if self.text_rewriter is None:
                print("Loading Qwen model...")
                import gc
                gc.collect()
                self.text_rewriter = QwenRewriter(device=self.device)
            
            # Initialize loop variables
            iteration = 0
            best_reward = 0.0
            best_version = None
            all_experiences = []
            should_continue = True
            
            # Main feedback loop - continue until threshold met or max iterations
            while iteration < max_iterations and should_continue:
                iteration += 1
                print("\n" + "="*60)
                print(f"ITERATION {iteration}/{max_iterations}")
                print("="*60)
            
                # Step 2: Generate 5 rewritten versions
                print(f"\n[Step 2] Generating 5 rewritten versions (Iteration {iteration})...")
                print("Please wait...")
                
                try:
                    versions = self.text_rewriter.generate_multiple_versions(text, num_versions=5)
                except Exception as e:
                    print(f"Error generating versions: {e}")
                    import traceback
                    traceback.print_exc()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return {"success": False, "error": f"Failed to generate rewritten versions: {str(e)}"}
                
                # Cleanup after generation using utility
                cleanup_memory(self.device, verbose=False)
                
                if not versions or len(versions) == 0:
                    return {"success": False, "error": "Failed to generate rewritten versions"}
                
                # Step 3: Display options and let user select
                print(f"\n[Step 3] Please select one of the following rewritten versions (Iteration {iteration}):")
                print("-" * 60)
                for i, version in enumerate(versions, 1):
                    # Analyze each version
                    version_analysis = self.sentiment_analyzer.analyze_text(version)
                    sentiment_improvement = version_analysis['sentiment_score'] - sentiment_score
                    print(f"\nOption {i}:")
                    print(f"  Text: {version}")
                    print(f"  Sentiment Improvement: +{sentiment_improvement:.2f}")
                    print(f"  New Sentiment: {version_analysis['sentiment_score']:.2f} ({version_analysis['sentiment_label']})")
                
                # Get user selection
                while True:
                    try:
                        selection = input("\nEnter your choice (1, 2, 3, 4, or 5): ").strip()
                        if selection in ['1', '2', '3', '4', '5']:
                            selected_index = int(selection) - 1
                            selected_text = versions[selected_index]
                            break
                        else:
                            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                    except (ValueError, KeyboardInterrupt):
                        print("Invalid input. Please enter 1, 2, 3, 4, or 5.")
                
                print(f"\n✓ You selected Option {selection}: {selected_text}")
                
                # Step 4: Get user feedback
                print(f"\n[Step 4] Please provide feedback on the selected rewrite (Iteration {iteration}):")
                print("Rate the quality of the rewritten text:")
                print("1. Very poor (0.0)")
                print("2. Poor (0.25)")
                print("3. Fair (0.5)")
                print("4. Good (0.75)")
                print("5. Excellent (1.0)")
                
                while True:
                    try:
                        rating_input = input("\nEnter your rating (1-5): ").strip()
                        rating_map = {
                            '1': 0.0,
                            '2': 0.25,
                            '3': 0.5,
                            '4': 0.75,
                            '5': 1.0
                        }
                        if rating_input in rating_map:
                            user_rating = rating_map[rating_input]
                            break
                        else:
                            print("Invalid rating. Please enter 1, 2, 3, 4, or 5.")
                    except KeyboardInterrupt:
                        return {"success": False, "error": "User cancelled feedback"}
                
                print(f"✓ Feedback received: {user_rating:.2f}")
                
                # Step 5: Compute reward and check threshold
                print(f"\n[Step 5] Computing reward (Iteration {iteration})...")
                
                # Lazy load trainer if needed
                if self.trainer is None:
                    print("Initializing trainer...")
                    # Unload text_rewriter model to free memory before loading trainer
                    existing_tokenizer = None
                    if self.text_rewriter is not None:
                        print("Freeing text rewriter model memory...")
                        existing_tokenizer = self.text_rewriter.tokenizer  # Save tokenizer reference
                        # Clear the model to free memory
                        if hasattr(self.text_rewriter, 'model'):
                            del self.text_rewriter.model
                            self.text_rewriter.model = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    
                    # Cleanup before initializing trainer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    try:
                        self.trainer = TextRewritingTrainer(
                            device=self.device,
                            existing_tokenizer=existing_tokenizer  # Share tokenizer if available
                        )
                    except Exception as e:
                        print(f"Error initializing trainer: {e}")
                        import traceback
                        traceback.print_exc()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise
            
                # Analyze selected text
                selected_analysis = self.sentiment_analyzer.analyze_text(selected_text)
                
                # Compute reward
                reward, component_scores = self.trainer.reward_function.compute_reward(
                    text, selected_text, user_rating=user_rating
                )
                
                print(f"\n{'='*60}")
                print(f"Reward computed: {reward:.3f}")
                print("Reward components:")
                for k, v in component_scores.items():
                    print(f"  - {k}: {v:.3f}")
                print(f"{'='*60}")
                
                # Track best result
                if reward > best_reward:
                    best_reward = reward
                    best_version = selected_text
                
                # Check if threshold is met
                if reward >= reward_threshold:
                    print(f"\n{'='*60}")
                    print(f"✓ SUCCESS! Reward threshold met: {reward:.3f} >= {reward_threshold:.2f}")
                    print(f"{'='*60}")
                    # Still update model once more with this good result
                    experience = {
                        'original_text': text,
                        'rewritten_text': selected_text,
                        'reward': reward,
                        'component_scores': component_scores,
                        'user_rating': user_rating
                    }
                    all_experiences.append(experience)
                    print(f"\n[Step 6] Updating model weights with PPO (final update)...")
                    update_stats = self.trainer._update_policy([experience])
                    
                    # Update text rewriter with final model
                    if self.trainer.model_save_path:
                        print(f"✓ Final model saved to: {self.trainer.model_save_path}")
                    if self.trainer.model is not None:
                        self.text_rewriter.model = self.trainer.model.pretrained_model
                        self.text_rewriter.model.eval()
                        self.text_rewriter.tokenizer = self.trainer.tokenizer
                    
                    break
                
                # Create experience for PPO update
                experience = {
                    'original_text': text,
                    'rewritten_text': selected_text,
                    'reward': reward,
                    'component_scores': component_scores,
                    'user_rating': user_rating
                }
                all_experiences.append(experience)
                
                # Update model immediately with PPO
                print(f"\n[Step 6] Updating model weights with PPO (Iteration {iteration})...")
                update_stats = self.trainer._update_policy([experience])
                
                # Verify model was saved
                if self.trainer.model_save_path:
                    print(f"✓ Model saved to: {self.trainer.model_save_path}")
                else:
                    print("⚠️ Warning: Model path not set after update")
            
                # Reload text_rewriter with the updated model for next iteration
                # CRITICAL: Ensure model is properly reloaded before next iteration
                try:
                    if self.trainer is not None and self.trainer.model is not None:
                        print("Updating text rewriter with trained model...")
                        
                        # Ensure model is in eval mode and on correct device
                        self.trainer.model.eval()
                        
                        # Recreate QwenRewriter with updated model if it doesn't exist
                        if self.text_rewriter is None:
                            print("Reinitializing text rewriter with updated model...")
                            self.text_rewriter = QwenRewriter(device=self.device)
                        
                        # Update with trained model - use pretrained_model from value head wrapper
                        self.text_rewriter.model = self.trainer.model.pretrained_model
                        self.text_rewriter.model.eval()
                        self.text_rewriter.tokenizer = self.trainer.tokenizer
                        
                        # Verify model is properly loaded
                        if self.text_rewriter.model is None:
                            raise RuntimeError("Failed to load updated model into text rewriter")
                        
                        print("✓ Text rewriter updated successfully with trained weights")
                    
                    # Cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                except Exception as e:
                    print(f"⚠️ Warning: Error updating text rewriter: {e}")
                    print("Will attempt to continue with existing model...")
                    import traceback
                    traceback.print_exc()
                
                print(f"\nIteration {iteration} complete. Current reward: {reward:.3f} (threshold: {reward_threshold:.2f})")
                
                # Check if we should continue - FIXED: Properly check conditions
                should_continue = (reward < reward_threshold) and (iteration < max_iterations)
                
                if should_continue:
                    print(f"Continuing to iteration {iteration + 1}...")
                    # Ensure we actually continue by NOT breaking
                elif iteration >= max_iterations:
                    print(f"Maximum iterations ({max_iterations}) reached.")
                    should_continue = False
                else:
                    print(f"Reward threshold ({reward_threshold:.2f}) met!")
                    should_continue = False
            
            # Save final model after all iterations (if not already saved)
            if self.trainer.model_save_path:
                print(f"\n[Saving] Final model already saved to {self.trainer.model_save_path}")
            else:
                print("\n[Saving] Saving final model...")
                # Use the trainer's save method to ensure proper formatting
                try:
                    if all_experiences:
                        # Save with the last experience's metadata
                        self.trainer.save_model_after_update(all_experiences[-1:], {})
                    else:
                        print("No experiences to save")
                except Exception as e:
                    print(f"Warning: Could not save final model: {e}")
                    import traceback
                    traceback.print_exc()
            
            final_model_path = self.trainer.model_save_path if self.trainer.model_save_path else "Unknown"
            print(f"\n✓ Model weights updated. Final model path: {final_model_path}")
            
            # Return results
            final_reward = best_reward if best_reward > 0 else reward
            final_version = best_version if best_version else selected_text
            
            return {
                "success": True,
                "original_text": text,
                "selected_version": final_version,
                "user_rating": user_rating,
                "reward": final_reward,
                "component_scores": component_scores,
                "original_analysis": original_analysis,
                "selected_analysis": selected_analysis,
                "iterations_completed": iteration,
                "threshold_met": final_reward >= reward_threshold,
                "all_experiences": all_experiences
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}"
            }
    
    def train_model(self, input_texts: List[str], resume_from: Optional[str] = None) -> dict:
        """
        Train the model with RLHF
        
        Args:
            input_texts: List of input texts to train on
            resume_from: Optional path to resume from
            
        Returns:
            Training results
        """
        try:
            if not input_texts:
                return {"error": "No input texts provided"}
            
            # Lazy load trainer only when needed for training
            if self.trainer is None:
                print("Initializing trainer for PPO training...")
                self.trainer = TextRewritingTrainer(device=self.device)
            
            print(f"Starting training with {len(input_texts)} input texts")
            
            # Start training
            final_model_path = self.trainer.train(input_texts, resume_from)
            
            # Get training summary
            summary = self.trainer.get_training_summary()
            
            return {
                "success": True,
                "final_model_path": final_model_path,
                "summary": summary,
                "episodes_completed": self.trainer.training_state['episode'],
                "best_reward": self.trainer.training_state['best_reward']
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}"
            }
    
    def interactive_mode(self):
        """Run interactive mode for testing and training"""
        print("\n" + "="*60)
        print("TEXT REWRITING WITH RLHF - INTERACTIVE MODE")
        print("="*60)
        print("Commands:")
        print("1. analyze <text> - Analyze text sentiment, tone, intent, meaning")
        print("2. rewrite <text> - Rewrite text (non-offensive, preserves meaning)")
        print("3. process <text> or feedback <text> - Complete workflow:")
        print("   - Analyze sentiment (if negative)")
        print("   - Generate 5 rewritten versions")
        print("   - Select best version")
        print("   - Provide feedback")
        print("   - Update model with PPO")
        print("4. train - Start training mode")
        print("5. list_models - List saved models")
        print("6. load_model <path> - Load a saved model")
        print("7. quit - Exit the application")
        print("="*60)
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("Goodbye!")
                    break
                
                elif command.startswith("analyze "):
                    text = command[8:].strip()
                    if text:
                        result = self.analyze_text(text)
                        if result.get("success"):
                            print(result["summary"])
                        else:
                            print(f"Error: {result.get('error')}")
                    else:
                        print("Please provide text to analyze")
                
                elif command.startswith("rewrite "):
                    text = command[8:].strip()
                    if text:
                        result = self.rewrite_text(text)
                        if result.get("success"):
                            print(f"\nOriginal: {result['original_text']}")
                            print(f"Rewritten: {result['rewritten_text']}")
                            print(f"Sentiment improvement: {result['improvement']['sentiment']:.3f}")
                            print(f"Tone improvement: {result['improvement']['tone']:.3f}")
                        else:
                            print(f"Error: {result.get('error')}")
                    else:
                        print("Please provide text to rewrite")
                
                elif command.startswith("process ") or command.startswith("feedback "):
                    # New command: process with selection and feedback
                    text = command.split(" ", 1)[1] if " " in command else None
                    if not text:
                        text = input("Enter text to process: ").strip()
                    
                    if text:
                        result = self.process_with_selection_and_feedback(text)
                        if result.get("success"):
                            print("\n" + "="*60)
                            print("PROCESSING COMPLETE")
                            print("="*60)
                            print(f"Original: {result['original_text']}")
                            print(f"Selected version: {result['selected_version']}")
                            print(f"User rating: {result['user_rating']:.2f}")
                            print(f"Reward: {result['reward']:.3f}")
                            print("✓ Model has been updated with your feedback!")
                        else:
                            print(f"Error: {result.get('error', result.get('message', 'Unknown error'))}")
                    else:
                        print("Please provide text to process")
                
                elif command == "train":
                    print("\nTraining mode activated!")
                    print("Enter texts to train on (one per line, empty line to finish):")
                    
                    training_texts = []
                    while True:
                        text = input("Training text: ").strip()
                        if not text:
                            break
                        training_texts.append(text)
                    
                    if training_texts:
                        print(f"\nStarting training with {len(training_texts)} texts...")
                        # Lazy load trainer
                        if self.trainer is None:
                            print("Initializing trainer...")
                            self.trainer = TextRewritingTrainer(device=self.device)
                        result = self.train_model(training_texts)
                        if result.get("success"):
                            print("Training completed successfully!")
                            print(result["summary"])
                        else:
                            print(f"Training failed: {result.get('error')}")
                    else:
                        print("No training texts provided")
                
                elif command == "list_models":
                    models = self.model_manager.list_models()
                    if models:
                        print(f"\nFound {len(models)} models:")
                        for i, model in enumerate(models, 1):
                            print(f"{i}. {model.get('model_name', 'Unknown')} - {model.get('timestamp', 'Unknown')}")
                    else:
                        print("No models found")
                
                elif command.startswith("load_model "):
                    model_path = command[11:].strip()
                    if model_path:
                        # Lazy load trainer if needed
                        if self.trainer is None:
                            self.trainer = TextRewritingTrainer(device=self.device)
                        if self.trainer.load_model(model_path):
                            # Update text_rewriter to use loaded model
                            self.text_rewriter.model = self.trainer.qwen_rewriter.model
                            self.text_rewriter.tokenizer = self.trainer.qwen_rewriter.tokenizer
                            print("Model loaded successfully!")
                        else:
                            print("Failed to load model")
                    else:
                        print("Please provide model path")
                
                else:
                    print("Unknown command. Type 'quit' to exit or see commands above.")
            
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Text Rewriting with RLHF")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to run on")
    parser.add_argument("--mode", default="process", choices=["interactive", "analyze", "rewrite", "train", "process", "feedback"],
                       help="Mode to run in (default: process - full workflow with user input)")
    parser.add_argument("--text", help="Text to analyze or rewrite")
    # style removed; rewrite is always non-offensive
    parser.add_argument("--input_file", help="File containing input texts for training")
    parser.add_argument("--resume_from", help="Path to resume training from")
    
    args = parser.parse_args()
    
    try:
        # Initialize app
        app = TextRewritingApp(device=args.device)
        
        if args.mode == "process" or args.mode == "feedback":
            # Default mode: Full workflow with user input
            # Get text from argument or prompt user
            if not args.text:
                print("\n" + "="*60)
                print("TEXT REWRITING WITH USER SELECTION & FEEDBACK")
                print("="*60)
                text = input("\nEnter text to rewrite: ").strip()
                if not text:
                    print("Error: No text provided")
                    return
            else:
                text = args.text
            
            result = app.process_with_selection_and_feedback(text)
            if result.get("success"):
                print("\n" + "="*60)
                print("PROCESSING COMPLETE")
                print("="*60)
                print(f"Original: {result['original_text']}")
                print(f"Selected version: {result['selected_version']}")
                print(f"User rating: {result['user_rating']:.2f}")
                print(f"Reward: {result['reward']:.3f}")
                print("✓ Model has been updated with your feedback!")
            else:
                print(f"Error: {result.get('error', result.get('message', 'Unknown error'))}")
        
        elif args.mode == "interactive":
            app.interactive_mode()
        
        elif args.mode == "analyze":
            if not args.text:
                print("Error: --text required for analyze mode")
                return
            
            result = app.analyze_text(args.text)
            if result.get("success"):
                print(result["summary"])
            else:
                print(f"Error: {result.get('error')}")
        
        elif args.mode == "rewrite":
            if not args.text:
                print("Error: --text required for rewrite mode")
                return
            
            result = app.rewrite_text(args.text)
            if result.get("success"):
                print(f"Original: {result['original_text']}")
                print(f"Rewritten: {result['rewritten_text']}")
                print(f"Sentiment improvement: {result['improvement']['sentiment']:.3f}")
                print(f"Tone improvement: {result['improvement']['tone']:.3f}")
            else:
                print(f"Error: {result.get('error')}")
        
        elif args.mode == "train":
            if not args.input_file:
                print("Error: --input_file required for train mode")
                return
            
            try:
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    input_texts = [line.strip() for line in f if line.strip()]
                
                if not input_texts:
                    print("Error: No valid texts found in input file")
                    return
                
                result = app.train_model(input_texts, args.resume_from)
                if result.get("success"):
                    print("Training completed successfully!")
                    print(result["summary"])
                else:
                    print(f"Training failed: {result.get('error')}")
            
            except FileNotFoundError:
                print(f"Error: Input file '{args.input_file}' not found")
            except Exception as e:
                print(f"Error reading input file: {e}")
    
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
