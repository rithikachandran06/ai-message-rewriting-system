"""
Qwen2.5-0.5B-Instruct Rewriter (CPU-friendly)
- Model: Qwen/Qwen2.5-0.5B-Instruct
- Prompt enforces politeness/positivity and preserves meaning/intent
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Optional
import sys
import os

# Windows-specific memory optimizations - set before torch import
if sys.platform == "win32":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = (
    """You are a text rewriting assistant. Rewrite the given text to be non-offensive, respectful, and neutral-to-positive.
Preserve the core intent and information without contradicting it. Do NOT flip the meaning with negations.
NEVER negate or contradict the original claim (avoid: 'not', 'never', "won't", "can't", "don't").
If the text is harmful or threatening, de-escalate by expressing the underlying feeling or concern in neutral terms,
without adding new information and without violent language.
Do NOT add offers, requests, apologies, promises, questions, or instructions. Output exactly ONE sentence.
Do NOT add: "please let me know", offers of help, or any questions.
Do NOT change the intent category of the original (e.g., question stays a question, request stays a request, a warning remains a warning).
Exception: For harmful or violent threats, you MAY use neutral deterrent phrases like "I won't hesitate to take appropriate action" or "Your actions will have consequences" or "I won't hesitate to deal with your actions" (one of these), which keep the deterrent intent without violent wording.
IMPORTANT: Do NOT respond to the text and do NOT add explanations.
Return ONLY the rewritten statement, concise and direct.

Original text: {text}
Rewritten text (non-offensive, no contradiction, one sentence):"""
)

class QwenRewriter:
    def __init__(self, device: str = "cpu", max_new_tokens: int = 96) -> None:
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Always use float32 on CPU (Windows has issues with float16)
        # Only use float16 if explicitly on CUDA and confirmed working
        model_dtype = torch.float32  # Use float32 for stability on Windows
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with memory optimization and safety checks
            # Use safe loading for Windows compatibility
            import gc
            gc.collect()
            
            # Try with safetensors first (better for Windows), fallback if not available
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, 
                    dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True  # Use safetensors for better Windows compatibility
                )
            except Exception:
                # Fallback without safetensors if not available
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, 
                    dtype=model_dtype,
                    low_cpu_mem_usage=True
                )
            self.model.to(self.device)
            self.model.eval()
            
            # Ensure model is properly initialized
            with torch.no_grad():
                # Test forward pass to ensure memory is valid
                dummy_input = self.tokenizer("test", return_tensors="pt").to(self.device)
                if dummy_input['input_ids'].numel() > 0:
                    _ = self.model(**dummy_input)
                    del dummy_input
            
            # Cleanup after loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading Qwen model: {e}")
            import gc
            gc.collect()
            raise

    def build_prompt(self, text: str) -> str:
        return PROMPT.format(text=text.strip())

    @torch.no_grad()
    def rewrite(self, text: str, temperature: float = 0.7, top_p: float = 0.9) -> str:
        try:
            prompt = self.build_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Ensure inputs are properly moved to device and contiguous
            inputs = {k: v.to(self.device).contiguous() if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Validate inputs
            if 'input_ids' not in inputs or inputs['input_ids'].numel() == 0:
                return text  # Return original if tokenization fails
            
            out = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # Ensure output is valid before decoding
            if out.numel() == 0:
                return text
            
            full = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Extract rewritten text - check multiple possible formats
            rewritten = None
            if "Rewritten text (same meaning, more polite):" in full:
                rewritten = full.split("Rewritten text (same meaning, more polite):", 1)[1].strip()
            elif "Rewritten text:" in full:
                rewritten = full.split("Rewritten text:", 1)[1].strip()
            elif "Rewritten:" in full:
                rewritten = full.split("Rewritten:", 1)[1].strip()
            else:
                # If no marker found, try to extract just the response part
                # Remove the prompt part if it's still there
                if prompt.lower() in full.lower():
                    # Find where prompt ends and response begins
                    prompt_end = full.lower().find(prompt.lower()) + len(prompt)
                    rewritten = full[prompt_end:].strip()
                else:
                    rewritten = full.strip()
            
            # Clean up: take first line/sentence only, remove extra explanations
            rewritten = rewritten.split("\n\n")[0].split("\n")[0].strip()
            # Remove any trailing question marks or explanations that might indicate the model is responding
            # Keep it concise
            if len(rewritten) > 200:  # If too long, it might be answering
                rewritten = rewritten[:200].rsplit('.', 1)[0] + '.'
            
            # Cleanup
            del inputs, out
            import gc
            gc.collect()
            
            return rewritten
        except Exception as e:
            print(f"Error in rewrite: {e}")
            import gc
            gc.collect()
            return text  # Return original on error
    
    def generate_multiple_versions(self, text: str, num_versions: int = 5) -> list:
        """
        Generate multiple rewritten versions with varying temperature for diversity
        
        Args:
            text: Input text to rewrite
            num_versions: Number of versions to generate
            
        Returns:
            List of rewritten texts
        """
        import gc
        import torch
        versions = []
        # Use different temperatures for diversity
        if num_versions == 5:
            temperatures = [0.5, 0.6, 0.7, 0.8, 0.9]
        elif num_versions == 3:
            temperatures = [0.5, 0.7, 0.9]
        else:
            temperatures = [0.7] * num_versions
        
        for i, temp in enumerate(temperatures[:num_versions]):
            version = self.rewrite(text, temperature=temp, top_p=0.9)
            if version and version not in versions:  # Avoid duplicates
                versions.append(version)
            
            # Cleanup after each generation to save memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # If we got less than requested, generate more with different temps
        while len(versions) < num_versions:
            temp = 0.6 + (len(versions) * 0.1)
            version = self.rewrite(text, temperature=temp, top_p=0.9)
            if version and version not in versions:
                versions.append(version)
            
            # Cleanup after each generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if len(versions) >= num_versions:
                break
        
        return versions[:num_versions]

