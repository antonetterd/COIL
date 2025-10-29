import time
import joblib
import onnxruntime as ort
import numpy as np

def benchmark_models():
    """Compare performance between ONNX and Pickle models"""
    
    # Load both models
    svm_model = joblib.load('svm_cry_classifier_3class.pkl')
    onnx_session = ort.InferenceSession('svm_cry_classifier.onnx')
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    # Create dummy features
    dummy_features = np.random.random((1, 384)).astype(np.float32)
    
    # Benchmark ONNX
    onnx_times = []
    for _ in range(1000):
        start = time.time()
        onnx_session.run([output_name], {input_name: dummy_features})
        onnx_times.append(time.time() - start)
    
    # Benchmark Pickle
    pickle_times = []
    for _ in range(1000):
        start = time.time()
        svm_model.predict_proba(dummy_features)
        pickle_times.append(time.time() - start)
    
    print(f"ONNX - Avg: {np.mean(onnx_times)*1000:.3f}ms, Min: {np.min(onnx_times)*1000:.3f}ms")
    print(f"Pickle - Avg: {np.mean(pickle_times)*1000:.3f}ms, Min: {np.min(pickle_times)*1000:.3f}ms")
    print(f"Speedup: {np.mean(pickle_times)/np.mean(onnx_times):.2f}x")

if __name__ == "__main__":
    benchmark_models()