"""
Qwen2.5-0.5B-Instruct Live Human Feedback PPO Update
- Usage (non-interactive rating):
  python3 qwen_live_feedback.py --text "your text here" --rating 4
- Saves/loads weights from ./qwen_ppo
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
from typing import Optional
import torch
from datasets import Dataset
from transformers import AutoTokenizer
try:
    from trl.experimental.ppo import (
        AutoModelForCausalLMWithValueHead,
        PPOConfig,
        PPOTrainer,
    )
except ImportError:
    from trl import PPOConfig, PPOTrainer
    from trl.models import AutoModelForCausalLMWithValueHead

from qwen_rewriter import MODEL_NAME, PROMPT
from sentiment_analyzer import SentimentAnalyzer
from reward_function import RewardFunction


def map_star_to_scalar(star: int) -> float:
    star = max(1, min(5, star))
    return {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}[star]


def main():
    parser = argparse.ArgumentParser(description="Qwen live PPO update with human feedback")
    parser.add_argument("--text", required=True, help="Input text to rewrite")
    parser.add_argument("--rating", type=int, required=True, help="Human rating 1-5")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--save_dir", default="./qwen_ppo", help="Where to save updated weights")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)

    load_path = args.save_dir if os.path.exists(os.path.join(args.save_dir, "config.json")) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(load_path, dtype=torch.float32)
    model.to(device)

    config = PPOConfig(
        model_name=load_path,
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        ppo_epochs=1,
        gradient_accumulation_steps=1,
        target_kl=0.1,
        adap_kl_ctrl=True,
        optimize_cuda_cache=False,
        log_with=None,
    )

    query = PROMPT.format(text=args.text.strip())
    dataset = Dataset.from_dict({"query": [query]})
    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer, dataset=dataset)

    # Tokenize prompt
    enc = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = enc["input_ids"]
    input_len = input_ids.shape[1]

    # Generate response
    with torch.no_grad():
        generated = model.generate(
            **enc,
            do_sample=True,
            max_new_tokens=96,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Extract response tokens after the prompt
    resp_only = generated[:, input_len:]
    if resp_only.numel() == 0:
        resp_only = generated[:, -1:]

    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    if "Rewritten:" in full_text:
        rewritten = full_text.split("Rewritten:", 1)[1].strip()
    else:
        rewritten = full_text.strip()
    rewritten = rewritten.split("\n\n")[0].split("\n")[0].strip()

    print("\nOriginal:", args.text)
    print("Rewritten:", rewritten)

    # Rating to scalar
    rating_scalar = map_star_to_scalar(args.rating)

    # Compute reward
    analyzer = SentimentAnalyzer(device=device)
    reward_fn = RewardFunction(analyzer)
    reward_value, comp = reward_fn.compute_reward(args.text, rewritten, user_rating=rating_scalar)

    print("\nReward components:")
    for k, v in comp.items():
        print(f"- {k}: {v:.3f}")

    reward_tensor = torch.tensor([reward_value], dtype=torch.float32).to(device)

    # PPO update
    stats = trainer.step([input_ids[0]], [resp_only[0]], [reward_tensor])

    # Save updated policy
    trainer.model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"\nModel updated and saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
