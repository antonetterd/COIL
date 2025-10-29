import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import glob
import os

def find_latest_model():
    """Find the latest SVM model file"""
    model_files = glob.glob('svm_cry_classifier_*.pkl')
    if not model_files:
        raise FileNotFoundError("No SVM model files found! Run train_model.py first.")
    
    # Get the latest model by modification time
    latest_model = sorted(model_files, key=os.path.getmtime)[-1]
    print(f"Found model: {latest_model}")
    
    # Find corresponding scaler
    model_name = os.path.basename(latest_model)
    if '3class' in model_name:
        scaler_pattern = 'scaler_3class_*.pkl'
    else:
        scaler_pattern = 'scaler_*.pkl'
    
    scaler_files = glob.glob(scaler_pattern)
    if not scaler_files:
        # Try to find any scaler
        scaler_files = glob.glob('scaler_*.pkl')
    
    if scaler_files:
        scaler_path = sorted(scaler_files, key=os.path.getmtime)[-1]
        print(f"Found scaler: {scaler_path}")
    else:
        scaler_path = None
        print("Warning: No scaler file found!")
    
    return latest_model, scaler_path

def convert_svm_to_onnx(svm_model_path, scaler_path, output_path):
    """Convert SVM model and scaler to ONNX format"""
    
    # Load the original model and scaler
    svm_model = joblib.load(svm_model_path)
    
    if scaler_path:
        scaler = joblib.load(scaler_path)
        # Save scaler info for reference
        with open('scaler_info.txt', 'w') as f:
            f.write(f"Scaler: {os.path.basename(scaler_path)}\n")
            f.write(f"Features: {scaler.n_features_in_}\n")
    else:
        scaler = None
    
    # Get feature dimension from the model
    n_features = svm_model.n_features_in_
    
    # Convert scaler first
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert SVM model to ONNX
    onnx_model = convert_sklearn(svm_model, initial_types=initial_type)
    
    # Save ONNX model
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f" ONNX model saved to: {output_path}")
    print(f" Input shape: [batch_size, {n_features}]")
    print(f" Model type: {type(svm_model).__name__}")
    print(f" Number of classes: {len(svm_model.classes_)}")
    
    return n_features

if __name__ == "__main__":
    try:
        # Automatically find the latest model
        svm_path, scaler_path = find_latest_model()
        
        # Create ONNX filename based on original model name
        base_name = os.path.splitext(os.path.basename(svm_path))[0]
        onnx_output = f"{base_name}.onnx"
        
        n_features = convert_svm_to_onnx(svm_path, scaler_path, onnx_output)
        
        print(f"\n🎉 Conversion successful!")
        print(f" Original: {os.path.basename(svm_path)}")
        print(f" ONNX: {onnx_output}")
        if scaler_path:
            print(f" Scaler: {os.path.basename(scaler_path)}")
        
    except Exception as e:
        print(f" Conversion failed: {e}")
        print("\n Available files in directory:")
        files = glob.glob('*.pkl') + glob.glob('*.json')
        for file in files:
            print(f"   - {file}")