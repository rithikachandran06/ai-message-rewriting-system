"""
Quick Start Script for Text Rewriting with RLHF
Demonstrates the system with a simple example
"""

from main import TextRewritingApp

def quick_demo():
    """Run a quick demonstration of the system"""
    print("="*60)
    print("TEXT REWRITING WITH RLHF - QUICK DEMO")
    print("="*60)
    
    # Initialize the app
    print("Initializing system...")
    app = TextRewritingApp(device="auto")
    print("System ready!\n")
    
    # Example texts to demonstrate
    example_texts = [
        "This is terrible and I hate it",
        "You are stupid and wrong",
        "I need help with this problem",
        "This doesn't work at all",
        "I'm frustrated with this service"
    ]
    
    print("Demonstrating text analysis and rewriting:")
    print("-" * 40)
    
    for i, text in enumerate(example_texts, 1):
        print(f"\n{i}. Original text: '{text}'")
        
        # Analyze the text
        analysis = app.analyze_text(text)
        if analysis.get("success"):
            print(f"   Analysis: {analysis['analysis']['sentiment_label']} sentiment, "
                  f"{analysis['analysis']['tone_label']} tone, "
                  f"{analysis['analysis']['intent_label']} intent")
        
        # Rewrite the text
        rewrite_result = app.rewrite_text(text, "polite")
        if rewrite_result.get("success"):
            print(f"   Rewritten: '{rewrite_result['rewritten_text']}'")
            print(f"   Improvement: sentiment +{rewrite_result['improvement']['sentiment']:.2f}, "
                  f"tone +{rewrite_result['improvement']['tone']:.2f}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo start training with your own texts:")
    print("1. Run: python main.py --mode interactive")
    print("2. Use the 'train' command")
    print("3. Enter your texts and rate the rewrites")
    print("\nOr use command line:")
    print("python main.py --mode train --input_file sample_inputs.txt")

if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
