# Text Rewriting with RLHF (Reinforcement Learning from Human Feedback)

A comprehensive deep learning system that analyzes text sentiment, tone, intent, and meaning, then rewrites text to be more polite, positive, and friendly using Reinforcement Learning from Human Feedback (RLHF) with PPO (Proximal Policy Optimization).

## Features

- **Sentiment Analysis**: Analyzes sentiment, tone, intent, and meaning clarity
- **Text Rewriting**: Rewrites text to be more polite, positive, or friendly
- **RLHF Training**: Uses PPO algorithm to improve rewriting based on user feedback
- **Memory Optimized**: Designed for 8GB RAM systems
- **Model Persistence**: Save and load trained models
- **Interactive Mode**: Command-line interface for testing and training

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
```

## Usage

### Interactive Mode (Recommended)
```bash
python main.py --mode interactive
```

### Command Line Usage

**Analyze text:**
```bash
python main.py --mode analyze --text "This is terrible and I hate it"
```

**Rewrite text:**
```bash
python main.py --mode rewrite --text "This is terrible and I hate it" --style polite
```

**Train model:**
```bash
python main.py --mode train --input_file sample_inputs.txt
```

**Resume training:**
```bash
python main.py --mode train --input_file sample_inputs.txt --resume_from ./models/model_path
```

## Interactive Commands

When in interactive mode, you can use these commands:

- `analyze <text>` - Analyze text sentiment, tone, intent, meaning
- `rewrite <text> [style]` - Rewrite text (polite/positive/friendly)
- `train` - Start training mode
- `list_models` - List saved models
- `load_model <path>` - Load a saved model
- `quit` - Exit the application

## System Architecture

### Components

1. **SentimentAnalyzer** (`sentiment_analyzer.py`)
   - Analyzes sentiment, tone, intent, and meaning
   - Uses VADER sentiment analysis and TextBlob
   - Provides comprehensive text analysis

2. **TextRewriter** (`text_rewriter.py`)
   - Policy network for text rewriting
   - Rule-based fallback system
   - Generates polite, positive, and friendly text

3. **PPOTrainer** (`ppo_trainer.py`)
   - Implements PPO algorithm for RLHF
   - Manages experience buffer and policy updates
   - Handles memory optimization

4. **RewardFunction** (`reward_function.py`)
   - Computes rewards based on user ratings
   - Evaluates text quality improvements
   - Combines multiple quality metrics

5. **ModelManager** (`model_manager.py`)
   - Handles model saving and loading
   - Manages model versioning
   - Provides model cleanup utilities

6. **TextRewritingTrainer** (`main_training_system.py`)
   - Orchestrates the entire training pipeline
   - Manages memory usage for 8GB RAM systems
   - Handles training state and stopping criteria

## Training Process

1. **Input Analysis**: Analyze input text for sentiment, tone, intent, and meaning
2. **Text Rewriting**: Generate rewritten version using policy network
3. **User Rating**: Get user feedback on the rewritten text
4. **Reward Computation**: Calculate reward based on user rating and quality metrics
5. **Policy Update**: Use PPO to update the policy network
6. **Model Saving**: Save updated model periodically

## Memory Optimization

The system is optimized for 8GB RAM systems:

- Small batch sizes (8 samples)
- Frequent memory cleanup
- Gradient accumulation
- Model checkpointing
- Efficient data structures

## Configuration

Key training parameters in `main_training_system.py`:

```python
self.training_config = {
    'max_episodes': 100,
    'min_reward_threshold': 0.8,
    'patience': 10,
    'batch_size': 8,
    'update_frequency': 5,
    'save_frequency': 10,
    'memory_cleanup_frequency': 5
}
```

## Model Files

Models are saved in the `./models/` directory with:
- `model.pt` - Model weights and optimizer state
- `metadata.json` - Training metadata
- `training_stats.pkl` - Training statistics

## Example Workflow

1. **Start the application:**
   ```bash
   python main.py --mode interactive
   ```

2. **Analyze some text:**
   ```
   analyze This is terrible and I hate it
   ```

3. **Rewrite the text:**
   ```
   rewrite This is terrible and I hate it polite
   ```

4. **Start training:**
   ```
   train
   ```
   Then enter training texts one by one.

5. **Rate the rewritten texts** when prompted (1-5 scale)

6. **Continue training** until the model reaches satisfactory performance

## Troubleshooting

### Memory Issues
- Reduce batch size in training config
- Increase memory cleanup frequency
- Use CPU instead of GPU if needed

### Model Loading Issues
- Check model file paths
- Ensure model compatibility
- Use `list_models` to see available models

### Training Issues
- Ensure input texts are diverse
- Provide consistent feedback
- Check reward thresholds

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 8GB RAM minimum
- CUDA support (optional, for GPU acceleration)

## License

This project is provided as-is for educational and research purposes.
