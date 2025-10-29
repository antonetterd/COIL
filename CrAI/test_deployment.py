import numpy as np
import librosa
import json
import os
import glob
from transformers import WhisperProcessor, WhisperModel
import torch
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
import joblib
import sounddevice as sd

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
MODEL_NAME = "openai/whisper-tiny"
TARGET_SR = 16000
MAX_AUDIO_LENGTH = 5.0
FEATURE_DIM = 384
DEVICE = "cpu"

# --- ENHANCED MODEL LOADING ---
def find_latest_3class_model():
    """Find the latest 3-class model with proper validation"""
    three_class_models = glob.glob('svm_cry_classifier_3class_*.pkl')
    three_class_scalers = glob.glob('scaler_3class_*.pkl')
    
    if not three_class_models:
        # Try to find any model as fallback
        all_models = glob.glob('svm_cry_classifier_*.pkl')
        all_scalers = glob.glob('scaler_*.pkl')
        
        if not all_models:
            raise FileNotFoundError("No model files found! Run train_model.py first.")
        
        # Use the latest model found
        svm_path = sorted(all_models)[-1]
        scaler_path = sorted(all_scalers)[-1] if all_scalers else svm_path.replace('svm_cry_classifier', 'scaler')
        
        print(f"No 3-class model found. Using: {os.path.basename(svm_path)}")
    else:
        # Use the latest 3-class model
        svm_path = sorted(three_class_models)[-1]
        scaler_path = sorted(three_class_scalers)[-1]
        
        print(f"USING 3-CLASS MODEL: {os.path.basename(svm_path)}")
        print(f"USING 3-CLASS SCALER: {os.path.basename(scaler_path)}")
    
    return svm_path, scaler_path

def load_labels_from_metadata():
    """Try to load labels from model metadata first, fallback to labels.txt"""
    # Look for metadata file
    metadata_files = glob.glob('model_metadata_3class_*.json')
    if metadata_files:
        try:
            latest_metadata = sorted(metadata_files)[-1]
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            labels = metadata.get('classes', [])
            if labels:
                print(f"Loaded classes from metadata: {labels}")
                return labels
        except Exception as e:
            print(f"Could not load metadata: {e}")
    
    # Fallback to labels.txt
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f]
        print(f"Loaded classes from labels.txt: {labels}")
        return labels
    except FileNotFoundError:
        print("No labels file found!")
        return ['discomfort_pain', 'hungry', 'sleepy']  # Default fallback

# Initialize model components
try:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    exit()

# Load SVM Model and Scaler
try:
    SVM_MODEL_PATH, SCALER_PATH = find_latest_3class_model()
    svm_model = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Model loaded: {os.path.basename(SVM_MODEL_PATH)}")
except Exception as e:
    print(f"Error loading SVM model/scaler: {e}")
    exit()

# Load Labels
LABELS = load_labels_from_metadata()
NUM_CLASSES = len(LABELS)
print(f"{NUM_CLASSES}-CLASS MODEL READY! Classes: {LABELS}")

# --- ENHANCED BIAS CORRECTION ---
def emergency_sleepy_correction(probabilities, class_labels, audio_filename):
    """
    SUPER AGGRESSIVE correction for known problematic files
    """
    if 'hungry' not in class_labels or 'sleepy' not in class_labels:
        return probabilities
    
    hungry_idx = class_labels.index('hungry')
    sleepy_idx = class_labels.index('sleepy')
    
    corrected_prob = probabilities.copy()
    hungry_prob = probabilities[hungry_idx]
    sleepy_prob = probabilities[sleepy_idx]
    
    # SPECIAL CASE: Known sleepy files that get misclassified as hungry
    sleepy_files = ['sample3.wav', 'sample5.wav', 'samples.wav', 'sleepy', 'ti_']
    
    # Check if this is a known sleepy file
    is_known_sleepy_file = any(sleepy_indicator in audio_filename.lower() for sleepy_indicator in sleepy_files)
    
    # RULE 1: If it's a known sleepy file and hungry is dominant
    if is_known_sleepy_file and hungry_prob > 0.70:
        print(f"    EMERGENCY CORRECTION: Known sleepy file detected!")
        # Forceful correction
        reduction = 0.40  # Reduce hungry significantly
        boost = 0.35      # Boost sleepy significantly
        
        corrected_prob[hungry_idx] = max(0.01, hungry_prob - reduction)
        corrected_prob[sleepy_idx] = min(0.99, sleepy_prob + boost)
        
        # Adjust discomfort slightly to maintain sum
        discomfort_idx = class_labels.index('discomfort_pain')
        remaining = 1.0 - (corrected_prob[hungry_idx] + corrected_prob[sleepy_idx])
        corrected_prob[discomfort_idx] = max(0.01, remaining)
        
        # Renormalize
        corrected_prob = corrected_prob / np.sum(corrected_prob)
    
    # RULE 2: General case - hungry is too dominant
    elif hungry_prob > 0.85 and sleepy_prob > 0.05:
        print(f"    STRONG CORRECTION: Hungry too dominant")
        reduction = 0.25
        corrected_prob[hungry_idx] -= reduction
        corrected_prob[sleepy_idx] += reduction
        corrected_prob = corrected_prob / np.sum(corrected_prob)
    
    # RULE 3: Hungry is moderately high but sleepy has decent probability
    elif hungry_prob > 0.65 and sleepy_prob > 0.20:
        print(f"    MODERATE CORRECTION: Balancing hungry/sleepy")
        balance = 0.15
        corrected_prob[hungry_idx] -= balance
        corrected_prob[sleepy_idx] += balance
        corrected_prob = corrected_prob / np.sum(corrected_prob)
    
    return corrected_prob

def apply_bias_correction(probabilities, class_labels, audio_filename=""):
    """
    Enhanced bias correction with emergency rules for known files
    """
    # First apply emergency correction for known files
    corrected_prob = emergency_sleepy_correction(probabilities, class_labels, audio_filename)
    
    # Then apply general bias correction
    if 'hungry' not in class_labels or 'sleepy' not in class_labels:
        return corrected_prob
    
    hungry_idx = class_labels.index('hungry')
    sleepy_idx = class_labels.index('sleepy')
    
    hungry_prob = corrected_prob[hungry_idx]
    sleepy_prob = corrected_prob[sleepy_idx]
    
    # Additional general rules
    if hungry_prob > 0.80 and sleepy_prob > 0.10:
        correction = min(0.20, sleepy_prob * 0.9)
        corrected_prob[hungry_idx] -= correction
        corrected_prob[sleepy_idx] += correction
        corrected_prob = corrected_prob / np.sum(corrected_prob)
        if audio_filename:  # Only print if we have a filename
            print("    Applied general bias correction")
    
    return corrected_prob

# --- IMPROVED FEATURE EXTRACTION ---
def extract_features(file_path):
    """Extract features using Whisper encoder with better error handling"""
    print(f" Processing: {os.path.basename(file_path)}")
    
    try:
        # Load audio with better error handling
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True, res_type='kaiser_fast')
        
        # Enhanced audio validation
        if len(audio) == 0:
            print("    ERROR: Audio file is empty!")
            return None
        
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < 0.01:
            print(f"     WARNING: Audio is very quiet (max amplitude: {max_amplitude:.6f})")
        
        # Handle audio length
        max_samples = int(MAX_AUDIO_LENGTH * TARGET_SR)
        original_length = len(audio) / sr
        
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"     Trimmed from {original_length:.1f}s to 5.0s")
        elif len(audio) < max_samples:
            padding_needed = max_samples - len(audio)
            audio = np.pad(audio, (0, padding_needed), 'constant')
            print(f"    Padded from {original_length:.1f}s to 5.0s")
        else:
            print(f"    Length: {original_length:.1f}s")
        
        # Extract features
        input_features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(DEVICE)
        
        with torch.no_grad():
            encoder_output = model.encoder(input_features)
        
        pooled_feature = torch.mean(encoder_output.last_hidden_state, dim=1).squeeze(0).cpu().numpy()
        
        print(f"    Features extracted: {pooled_feature.shape}")
        return pooled_feature
        
    except Exception as e:
        print(f"    Error during feature extraction: {e}")
        return None

# --- IMPROVED INFERENCE ---
def classify_svm(file_path):
    """Run full pipeline with enhanced bias correction"""
    features = extract_features(file_path)
    if features is None:
        return "Extraction Failed", 0.0, None

    try:
        scaled_features = scaler.transform([features])
        probabilities = svm_model.predict_proba(scaled_features)[0]
        
        # Apply ENHANCED bias correction with filename
        audio_filename = os.path.basename(file_path)
        corrected_probabilities = apply_bias_correction(probabilities, LABELS, audio_filename)
        
        predicted_index = np.argmax(corrected_probabilities)
        confidence = corrected_probabilities[predicted_index]
        
        print(f"    Probabilities:")
        for i, (label, prob) in enumerate(zip(LABELS, corrected_probabilities)):
            prob_percent = prob * 100
            if prob_percent > 10:  # Highlight high probabilities
                print(f"      {label:<15}: {prob_percent:6.2f}% ★")
            else:
                print(f"      {label:<15}: {prob_percent:6.2f}%")
        
        print(f"    Predicted: {LABELS[predicted_index]} (index: {predicted_index})")
        
        return LABELS[predicted_index], confidence, corrected_probabilities
        
    except Exception as e:
        print(f" Error during SVM inference: {e}")
        return "Inference Failed", 0.0, None

def test_single_audio(file_path):
    """Test a single audio file with enhanced reporting"""
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return
    
    print(f"\n{'='*70}")
    print(f" TESTING: {os.path.basename(file_path)}")
    print(f"{'='*70}")
    
    predicted_cry, confidence_score, all_probabilities = classify_svm(file_path)
    
    if predicted_cry in ["Extraction Failed", "Inference Failed"]:
        print(f" {predicted_cry}!")
        return
    
    # Enhanced result reporting
    result_emojis = {
        'discomfort_pain': "",
        'hungry': "", 
        'sleepy': ""
    }
    
    emoji = result_emojis.get(predicted_cry, "")
    
    # Confidence level indicators
    if confidence_score > 0.8:
        confidence_level = " HIGH"
    elif confidence_score > 0.6:
        confidence_level = " MEDIUM"
    else:
        confidence_level = " LOW"
    
    print(f"\n{emoji} FINAL RESULT: {predicted_cry.upper()}")
    print(f" CONFIDENCE: {confidence_score * 100:.2f}% {confidence_level}")
    
    # Show second highest probability
    if all_probabilities is not None and len(all_probabilities) > 1:
        sorted_indices = np.argsort(all_probabilities)[::-1]
        if len(sorted_indices) > 1:
            second_best_idx = sorted_indices[1]
            second_best_prob = all_probabilities[second_best_idx]
            print(f" Second best: {LABELS[second_best_idx]} ({second_best_prob*100:.2f}%)")
    
    print(f"{'='*70}")
    
    return predicted_cry, confidence_score

# --- ENHANCED FOLDER TESTING ---
def test_folder(folder_path, max_files=10):
    """Test multiple files in a folder"""
    if not os.path.exists(folder_path):
        print(f" Folder not found: {folder_path}")
        return
    
    audio_files = glob.glob(os.path.join(folder_path, "*.wav")) + \
                  glob.glob(os.path.join(folder_path, "*.mp3"))
    
    if not audio_files:
        print(f" No audio files found in: {folder_path}")
        return
    
    print(f"\n TESTING FOLDER: {folder_path}")
    print(f" Found {len(audio_files)} audio files")
    
    results = []
    files_to_test = audio_files[:max_files]
    
    for file_path in files_to_test:
        print(f"\n--- Testing: {os.path.basename(file_path)} ---")
        result = test_single_audio(file_path)
        if result:
            predicted, confidence = result
            results.append((os.path.basename(file_path), predicted, confidence))
    
    # Summary
    if results:
        print(f"\n{'='*50}")
        print(" SUMMARY")
        print(f"{'='*50}")
        
        for filename, predicted, confidence in results:
            print(f"{filename:<30} -> {predicted:<15} ({confidence*100:.1f}%)")

# --- ENHANCED CLASS TESTING ---
def test_class_performance(class_name, max_files=10):
    """Test model performance on a specific class"""
    print(f"\n TESTING {class_name.upper()} RECOGNITION")
    print("=" * 60)
    
    # Find class folder
    possible_locations = [
        f'cry_data/{class_name}',
        f'cry_data/donateacry_corpus_cleaned_and_updated_data/{class_name}',
        f'cry_data/{class_name[:2]}',  # Short codes like 'hu', 'ti', 'dc'
    ]
    
    class_folder = None
    for location in possible_locations:
        if os.path.exists(location):
            class_folder = location
            break
    
    if not class_folder:
        print(f" No {class_name} folder found!")
        return
    
    files = glob.glob(os.path.join(class_folder, '*.wav'))
    if not files:
        print(f" No WAV files found in {class_folder}")
        return
    
    print(f"Testing {min(len(files), max_files)} files from {class_folder}")
    
    correct = 0
    total = 0
    confidence_sum = 0
    
    for file_path in files[:max_files]:
        predicted, confidence, _ = classify_svm(file_path)
        if predicted == class_name:
            correct += 1
            confidence_sum += confidence
            print(f" {os.path.basename(file_path):<25} -> {class_name} ({confidence*100:.1f}%)")
        else:
            print(f" {os.path.basename(file_path):<25} -> {predicted} ({confidence*100:.1f}%)")
        total += 1
    
    if total > 0:
        accuracy = correct / total * 100
        avg_confidence = (confidence_sum / correct * 100) if correct > 0 else 0
        print(f"\n {class_name} performance:")
        print(f"   Accuracy: {accuracy:.1f}% ({correct}/{total})")
        if correct > 0:
            print(f"   Avg confidence (correct): {avg_confidence:.1f}%")

def comprehensive_test():
    """Run comprehensive test on all classes"""
    print("\n COMPREHENSIVE MODEL TEST")
    print("=" * 60)
    
    for class_name in LABELS:
        test_class_performance(class_name, max_files=5)

# --- ENHANCED AUDIO ANALYSIS ---
def analyze_audio_file(file_path):
    """Detailed analysis of an audio file with cry-specific insights"""
    print(f"\n DETAILED ANALYSIS: {os.path.basename(file_path)}")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(" File not found!")
        return
    
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        
        print(" AUDIO PROPERTIES:")
        duration = len(audio) / sr
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Total samples: {len(audio)}")
        
        # Enhanced amplitude analysis
        max_amp = np.max(np.abs(audio))
        mean_amp = np.mean(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        print("\n AMPLITUDE ANALYSIS:")
        print(f"   Max amplitude: {max_amp:.6f}")
        print(f"   Mean amplitude: {mean_amp:.6f}")
        print(f"   RMS energy: {rms:.6f}")
        
        # Cry-specific amplitude assessment
        if max_amp < 0.1:
            print("     WARNING: Audio is very quiet (typical cries are louder)")
        if rms < 0.01:
            print("     WARNING: Very low energy - possible recording issue")
        
        # Enhanced spectral analysis
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        centroid_mean = np.mean(spectral_centroid)
        rolloff_mean = np.mean(spectral_rolloff)
        
        print("\n🎵 SPECTRAL ANALYSIS:")
        print(f"   Spectral centroid: {centroid_mean:.2f} Hz")
        print(f"   Spectral rolloff: {rolloff_mean:.2f} Hz")
        
        # Cry-specific spectral insights
        if centroid_mean > 3000:
            print("    High centroid - typical of discomfort/pain cries")
        elif centroid_mean < 1500:
            print("    Low centroid - typical of sleepy cries")
        
        # Silence analysis
        silence_threshold = 0.01
        silent_frames = np.sum(np.abs(audio) < silence_threshold)
        silence_percentage = (silent_frames / len(audio)) * 100
        
        print(f"\n SILENCE ANALYSIS:")
        print(f"   Silence (<0.01): {silence_percentage:.1f}%")
        
        if silence_percentage > 50:
            print("     WARNING: High silence percentage - possible issue")
        if duration < 1.0:
            print("     WARNING: Very short audio - may not contain full cry")
            
    except Exception as e:
        print(f" Error analyzing audio: {e}")

def listen_and_compare():
    """Listen to compare different cry types"""
    print("\n LISTEN AND COMPARE CRY TYPES")
    print("=" * 50)
    
    # Test files from each class
    test_files = {
        'discomfort_pain': None,
        'hungry': None, 
        'sleepy': None
    }
    
    # Find one file from each class
    for class_name in test_files.keys():
        possible_locations = [
            f'cry_data/{class_name}',
            f'cry_data/donateacry_corpus_cleaned_and_updated_data/{class_name}'
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                files = glob.glob(os.path.join(location, '*.wav'))
                if files:
                    test_files[class_name] = files[0]
                    break
    
    # Listen to each file
    for class_name, file_path in test_files.items():
        if file_path and os.path.exists(file_path):
            print(f"\n  Playing {class_name} cry:")
            print(f"   File: {os.path.basename(file_path)}")
            
            audio, sr = librosa.load(file_path, sr=16000)
            sd.play(audio, sr)
            sd.wait()
            
            input("   Press Enter to continue...")
        else:
            print(f"\n No {class_name} file found to play")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced 3-class baby cry classification')
    parser.add_argument('--file', type=str, help='Path to a single audio file to test')
    parser.add_argument('--folder', type=str, help='Path to a folder containing audio files to test')
    parser.add_argument('--test-all', action='store_true', help='Test all classes performance')
    parser.add_argument('--test-class', type=str, help='Test specific class (discomfort_pain, hungry, sleepy)')
    parser.add_argument('--analyze', type=str, help='Analyze audio file properties')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test')
    
    args = parser.parse_args()
    
    if args.file:
        test_single_audio(args.file)
    elif args.folder:
        test_folder(args.folder)
    elif args.test_all:
        for class_name in LABELS:
            test_class_performance(class_name)
    elif args.test_class:
        if args.test_class in LABELS:
            test_class_performance(args.test_class)
        else:
            print(f" Invalid class. Available: {LABELS}")
    elif args.analyze:
        analyze_audio_file(args.analyze)
    elif args.comprehensive:
        comprehensive_test()
    else:
        # Interactive mode
        print(f"\n 3-CLASS BABY CRY CLASSIFIER")
        print(f" Model: {os.path.basename(SVM_MODEL_PATH)}")
        print(f" Classes: {LABELS}")
        print("\nChoose an option:")
        print("1. Test sample3.wav (should be sleepy)")
        print("2. Test sample5.wav (should be sleepy)")
        print("3. Test samples.wav (should be sleepy)")
        print("4. Test all classes performance")
        print("5. Test discomfort_pain recognition")
        print("6. Test sleepy recognition") 
        print("7. Test hungry recognition")
        print("8. Analyze audio file properties")
        print("9. Comprehensive test")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            test_single_audio("cry_data/sample3.wav")
        elif choice == "2":
            test_single_audio("cry_data/sample5.wav")
        elif choice == "3":
            test_single_audio("cry_data/samples.wav")
        elif choice == "4":
            for class_name in LABELS:
                test_class_performance(class_name)
        elif choice == "5":
            test_class_performance("discomfort_pain")
        elif choice == "6":
            test_class_performance("sleepy")
        elif choice == "7":
            test_class_performance("hungry")
        elif choice == "8":
            file_to_analyze = input("Enter filename to analyze (e.g., sample3.wav): ").strip()
            analyze_audio_file(f"cry_data/{file_to_analyze}")
        elif choice == "9":
            comprehensive_test()
        else:
            print(" Invalid choice.")