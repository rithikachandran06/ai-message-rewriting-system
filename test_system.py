"""
Test script to verify the system works correctly
"""

import sys
import traceback
from main import TextRewritingApp

def test_basic_functionality():
    """Test basic functionality of the system"""
    print("Testing Text Rewriting System...")
    
    try:
        # Initialize app
        print("1. Initializing app...")
        app = TextRewritingApp(device="cpu")  # Use CPU for testing
        print("✓ App initialized successfully")
        
        # Test text analysis
        print("\n2. Testing text analysis...")
        test_text = "This is terrible and I hate it"
        result = app.analyze_text(test_text)
        
        if result.get("success"):
            print("✓ Text analysis working")
            print(f"  Sentiment: {result['analysis']['sentiment_label']} ({result['analysis']['sentiment_score']:.2f})")
            print(f"  Tone: {result['analysis']['tone_label']} ({result['analysis']['tone_score']:.2f})")
            print(f"  Intent: {result['analysis']['intent_label']} ({result['analysis']['intent_score']:.2f})")
        else:
            print(f"✗ Text analysis failed: {result.get('error')}")
            return False
        
        # Test text rewriting
        print("\n3. Testing text rewriting...")
        result = app.rewrite_text(test_text, "polite")
        
        if result.get("success"):
            print("✓ Text rewriting working")
            print(f"  Original: {result['original_text']}")
            print(f"  Rewritten: {result['rewritten_text']}")
            print(f"  Sentiment improvement: {result['improvement']['sentiment']:.2f}")
        else:
            print(f"✗ Text rewriting failed: {result.get('error')}")
            return False
        
        # Test model manager
        print("\n4. Testing model manager...")
        models = app.model_manager.list_models()
        print(f"✓ Model manager working (found {len(models)} models)")
        
        print("\n✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        traceback.print_exc()
        return False

def test_training_system():
    """Test training system components"""
    print("\nTesting Training System Components...")
    
    try:
        # Initialize app
        app = TextRewritingApp(device="cpu")
        
        # Test training system initialization
        print("1. Testing training system initialization...")
        trainer = app.trainer
        print("✓ Training system initialized")
        
        # Test reward function
        print("2. Testing reward function...")
        from reward_function import RewardFunction
        reward_func = RewardFunction(app.sentiment_analyzer)
        
        # Test reward computation (without user input)
        original = "This is terrible"
        rewritten = "This could be improved"
        reward, components = reward_func.compute_reward(original, rewritten, user_rating=0.5)
        print(f"✓ Reward function working (reward: {reward:.2f})")
        
        # Test PPO trainer
        print("3. Testing PPO trainer...")
        ppo_trainer = app.trainer.ppo_trainer
        print("✓ PPO trainer initialized")
        
        print("✓ All training system tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training system test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("TEXT REWRITING SYSTEM - TEST SUITE")
    print("="*60)
    
    # Run basic functionality tests
    basic_success = test_basic_functionality()
    
    # Run training system tests
    training_success = test_training_system()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Basic Functionality: {'PASS' if basic_success else 'FAIL'}")
    print(f"Training System: {'PASS' if training_success else 'FAIL'}")
    
    if basic_success and training_success:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nTo start using the system, run:")
        print("python main.py --mode interactive")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
