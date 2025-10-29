import glob
import os

print("🔍 Checking for model files in current directory...")
print(f"📁 Current directory: {os.getcwd()}\n")

# Check for all relevant files
file_patterns = [
    'svm_cry_classifier_*.pkl',
    'scaler_*.pkl', 
    'model_metadata_*.json',
    'labels.txt'
]

for pattern in file_patterns:
    files = glob.glob(pattern)
    if files:
        print(f"📂 Found {pattern}:")
        for file in files:
            size = os.path.getsize(file)
            mtime = os.path.getmtime(file)
            print(f"   - {file} ({size} bytes, modified: {mtime})")
    else:
        print(f"❌ No files matching: {pattern}")
    print()