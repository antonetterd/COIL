import onnxruntime as ort
import numpy as np
import joblib

print(" DEBUGGING ONNX MODEL OUTPUTS...")

# Load ONNX model
session = ort.InferenceSession("svm_cry_classifier_3class_20251028_174843.onnx")

print(" ONNX Model Details:")
for i, input_info in enumerate(session.get_inputs()):
    print(f"  Input {i}: {input_info.name}, Shape: {input_info.shape}")

for i, output_info in enumerate(session.get_outputs()):
    print(f"  Output {i}: {output_info.name}, Type: {output_info.type}")

# Test with actual data
scaler = joblib.load("scaler_3class_20251028_174843.pkl")
dummy_input = np.random.random((1, 384)).astype(np.float32)
dummy_scaled = scaler.transform(dummy_input)

print(f"\n Running inference...")
results = session.run(None, {'float_input': dummy_scaled})

print(f"\n Raw Output Results:")
for i, result in enumerate(results):
    print(f"Output {i} ({session.get_outputs()[i].name}):")
    print(f"  Type: {type(result)}")
    if isinstance(result, np.ndarray):
        print(f"  Shape: {result.shape}")
        print(f"  Values: {result}")
    else:
        print(f"  Value: {result}")
        print(f"  Length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        # Try to inspect the sequence
        if hasattr(result, '__len__') and len(result) > 0:
            for j, item in enumerate(result):
                print(f"    Item {j}: {type(item)} = {item}")

print(f"\n Interpretation:")
# Output 0 is the predicted label
predicted_label_idx = results[0][0]
labels = ['discomfort_pain', 'hungry', 'sleepy']
print(f"Predicted label index: {predicted_label_idx}")
print(f"Predicted label: {labels[predicted_label_idx]}")

# Output 1 is the probability sequence
if len(results) > 1:
    prob_output = results[1]
    print(f"Probability output type: {type(prob_output)}")
    if isinstance(prob_output, list) and len(prob_output) > 0:
        print("Probability sequence details:")
        for i, item in enumerate(prob_output):
            if hasattr(item, 'shape'):
                print(f"  Sequence item {i}: shape={item.shape}")
            else:
                print(f"  Sequence item {i}: {item}")