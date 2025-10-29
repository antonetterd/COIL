# check_training_needed.py
import glob
import os

print("🔍 CHECKING IF 3-CLASS TRAINING IS NEEDED:")
print("=" * 50)

# Check for 3-class models
three_class_models = glob.glob('svm_cry_classifier_3class_*.pkl')
three_class_scalers = glob.glob('scaler_3class_*.pkl')

if three_class_models and three_class_scalers:
    print("✅ 3-class models found:")
    print(f"  Model: {three_class_models[0]}")
    print(f"  Scaler: {three_class_scalers[0]}")
    
    # Check labels
    if os.path.exists('labels.txt'):
        with open('labels.txt', 'r') as f:
            labels = f.read().splitlines()
            print(f"  Labels: {labels}")
            
        if len(labels) == 3 and 'discomfort_pain' in labels:
            print("🎯 Perfect! 3-class setup is ready.")
        else:
            print("⚠️ Labels don't match expected 3-class structure.")
else:
    print("❌ No 3-class models found!")
    print("💡 You need to run: python train_model.py")
    
print(f"\n📊 Available models:")
svm_files = glob.glob('svm_cry_classifier_*.pkl')
for f in svm_files:
    print(f"  - {f}")