#!/usr/bin/env python3
"""
Train the ML model
"""

import sys
import os
from ml_model import expense_classifier

def main():
    print("=" * 60)
    print("🤖 EXPENSE CATEGORIZER ML MODEL TRAINING")
    print("=" * 60)
    
    # Create directory for model
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    
    print("\n🚀 Starting training process...")
    
    try:
        # Train the model
        accuracy = expense_classifier.train_model()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print(f"📊 Final Accuracy: {accuracy:.2%}")
        
        # Test with examples
        print("\n🧪 TEST PREDICTIONS:")
        print("-" * 40)
        
        test_cases = [
            "going to ambala",
            "pizza at dominos with friends",
            "monthly electricity bill payment",
            "buying new clothes at mall",
            "movie tickets for avengers",
            "uber ride to airport",
            "groceries from supermarket",
            "train ticket to delhi",
            "netflix subscription renewal",
            "dinner at restaurant"
        ]
        
        for test in test_cases:
            result = expense_classifier.predict(test)
            confidence_percent = result['confidence'] * 100
            
            # Color code based on confidence
            if confidence_percent > 90:
                conf_color = "🟢"
            elif confidence_percent > 70:
                conf_color = "🟡"
            else:
                conf_color = "🔴"
            
            print(f"\n📝 Input: '{test}'")
            print(f"   📍 Category: {result['category'].upper()}")
            print(f"   {conf_color} Confidence: {confidence_percent:.1f}%")
            print(f"   🤖 Model: {result['model_used']}")
            print(f"   💡 Explanation: {result['explanation']}")
        
        print("\n" + "=" * 60)
        print("🎉 MODEL IS READY TO USE!")
        print("👉 Run: python app.py")
        print("👉 Then open: http://localhost:5000")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())