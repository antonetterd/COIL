import os
import glob
import numpy as np
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import joblib
import json
import random
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Suppress warnings during librosa loading
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
DATA_DIR = 'cry_data'
MODEL_NAME = "openai/whisper-tiny"
CLASSES = {
    'hu': 'hungry',
    'dc': 'discomfort_pain',
    'ti': 'sleepy',
    'bp': 'discomfort_pain',
    'sc': 'discomfort_pain',      # scared
    'ch': 'discomfort_pain',      # cold/hot
    'lo': 'discomfort_pain',      # lonely
    'pa': 'discomfort_pain',      # pain
    'tired': 'sleepy',            # tired = sleepy
    'fatigue': 'sleepy',          # fatigue = sleepy
    'discomfort_pain': 'discomfort_pain',
    'hungry': 'hungry',
    'sleepy': 'sleepy',
    'scared': 'discomfort_pain',
    'cold': 'discomfort_pain',
    'hot': 'discomfort_pain',
    'lonely': 'discomfort_pain'
}
TARGET_SR = 16000
MAX_AUDIO_LENGTH = 5.0
FEATURE_DIM = 384
TARGET_SAMPLES_PER_CLASS = 1000  # All classes will have exactly this many samples

# Global cache for augmented audio
AUGMENTED_AUDIO_CACHE = {}

# --- DEVICE SETUP ---
device = "cpu"
print(f"Using device: {device}")

# Initialize Whisper components
try:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperModel.from_pretrained(MODEL_NAME)
    model.to(device)
except Exception as e:
    print(f"Failed to load Whisper model components. Error: {e}")
    exit()

# --- ENHANCED AUGMENTATION FUNCTIONS ---
def add_white_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio.astype(np.float32)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=1.2):
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)

def volume_shift(audio, volume_factor=1.2):
    return audio * volume_factor

def random_equalizer(audio, sr):
    """Apply random EQ changes to simulate different recording conditions"""
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    fft = np.fft.rfft(audio)
    
    bass_boost = np.random.uniform(0.8, 1.2)
    mid_boost = np.random.uniform(0.9, 1.1)
    treble_boost = np.random.uniform(0.95, 1.05)
    
    for i, freq in enumerate(freqs):
        if freq < 250:
            fft[i] *= bass_boost
        elif freq < 2000:
            fft[i] *= mid_boost
        else:
            fft[i] *= treble_boost
    
    return np.fft.irfft(fft).astype(np.float32)

def background_noise(audio, noise_level=0.01):
    background = np.random.normal(0, noise_level, len(audio))
    return (audio + background).astype(np.float32)

# --- AGGRESSIVE AUGMENTATION FOR ALL CLASSES ---
def aggressive_augmentation(audio, sr, num_variants=10):
    """Create multiple augmented versions of audio"""
    augmentations = []
    
    # Base augmentations
    augmentations.append(audio)  # Keep original
    
    # Noise variations
    augmentations.append(add_white_noise(audio, 0.003))
    augmentations.append(add_white_noise(audio, 0.006))
    augmentations.append(background_noise(audio, 0.01))
    
    # Pitch variations
    augmentations.append(pitch_shift(audio, sr, -2))
    augmentations.append(pitch_shift(audio, sr, -1))
    augmentations.append(pitch_shift(audio, sr, 1))
    augmentations.append(pitch_shift(audio, sr, 2))
    
    # Time stretching
    augmentations.append(time_stretch(audio, 0.8))
    augmentations.append(time_stretch(audio, 0.9))
    augmentations.append(time_stretch(audio, 1.1))
    augmentations.append(time_stretch(audio, 1.2))
    
    # Volume variations
    augmentations.append(volume_shift(audio, 0.6))
    augmentations.append(volume_shift(audio, 0.8))
    augmentations.append(volume_shift(audio, 1.3))
    augmentations.append(volume_shift(audio, 1.5))
    
    # Combined effects
    audio_combo1 = time_stretch(audio, 0.85)
    audio_combo1 = pitch_shift(audio_combo1, sr, -1)
    audio_combo1 = volume_shift(audio_combo1, 0.9)
    augmentations.append(audio_combo1)
    
    audio_combo2 = time_stretch(audio, 1.15)
    audio_combo2 = pitch_shift(audio_combo2, sr, 1)
    audio_combo2 = add_white_noise(audio_combo2, 0.004)
    augmentations.append(audio_combo2)
    
    return augmentations[:num_variants]

# --- UTILITY FUNCTIONS ---
def extract_label_from_filename(filename):
    try:
        # Handle augmented files
        if filename.startswith("aug_"):
            return filename.split('_')[1]  # aug_hungry_123 -> hungry
            
        basename = os.path.basename(filename).lower()
        
        # Check for class codes in filename
        for code in CLASSES.keys():
            if f'_{code}' in basename or f'-{code}' in basename or f'{code}.' in basename:
                return CLASSES[code]
        
        # Check if class name is in filename
        for class_name in ['hungry', 'discomfort_pain', 'sleepy', 'discomfort', 'pain', 'scared', 'cold', 'hot', 'lonely', 'tired', 'fatigue']:
            if class_name in basename:
                if class_name in ['discomfort', 'pain', 'scared', 'cold', 'hot', 'lonely']:
                    return 'discomfort_pain'
                elif class_name in ['tired', 'fatigue']:
                    return 'sleepy'
                return class_name
        
        # Check folder structure
        folder_name = os.path.basename(os.path.dirname(filename)).lower()
        if folder_name in CLASSES:
            return CLASSES[folder_name]
        elif folder_name in ['discomfort', 'pain', 'scared', 'cold', 'hot', 'lonely']:
            return 'discomfort_pain'
        elif folder_name in ['tired', 'fatigue']:
            return 'sleepy'
            
        return None
    except:
        return None

def process_audio_and_extract_features_enhanced(file_path, apply_augment=False, augmentation_type=None):
    """Enhanced version that can handle in-memory augmented audio"""
    try:
        # Check if this is an augmented audio from our cache
        if file_path in AUGMENTED_AUDIO_CACHE:
            audio, sr = AUGMENTED_AUDIO_CACHE[file_path]
        else:
            # Regular file path - load from disk
            audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True, res_type='kaiser_fast')

        # Apply augmentation if requested
        if apply_augment and augmentation_type and file_path not in AUGMENTED_AUDIO_CACHE:
            if augmentation_type == 'noise':
                audio = add_white_noise(audio, noise_factor=random.uniform(0.002, 0.01))
            elif augmentation_type == 'pitch_up':
                audio = pitch_shift(audio, sr, n_steps=random.choice([1, 2, 3]))
            elif augmentation_type == 'pitch_down':
                audio = pitch_shift(audio, sr, n_steps=random.choice([-1, -2, -3]))
            elif augmentation_type == 'time_stretch_fast':
                audio = time_stretch(audio, rate=random.uniform(1.1, 1.3))
            elif augmentation_type == 'time_stretch_slow':
                audio = time_stretch(audio, rate=random.uniform(0.7, 0.9))
            elif augmentation_type == 'volume_up':
                audio = volume_shift(audio, volume_factor=random.uniform(1.1, 1.5))
            elif augmentation_type == 'volume_down':
                audio = volume_shift(audio, volume_factor=random.uniform(0.5, 0.9))

        max_samples = int(MAX_AUDIO_LENGTH * TARGET_SR)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            padding_needed = max_samples - len(audio)
            audio = np.pad(audio, (0, padding_needed), 'constant')

        input_features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(device)

        with torch.no_grad():
            encoder_output = model.encoder(input_features)

        pooled_feature = torch.mean(encoder_output.last_hidden_state, dim=1).squeeze(0).cpu().numpy()
        return pooled_feature

    except Exception as e:
        print(f"\n[ERROR] Could not process file: {file_path}. Error: {e}")
        return None

# --- EQUAL AUGMENTATION FOR ALL CLASSES ---
def augment_class_files(class_files, target_samples=1000):
    """Augment each class to have exactly target_samples number of files"""
    print(f"\n AUGMENTING ALL CLASSES TO {target_samples} SAMPLES EACH")
    
    augmented_files = []
    augmented_labels = []
    
    for class_name, files in class_files.items():
        print(f"\nProcessing {class_name}: {len(files)} → {target_samples}")
        
        if len(files) == 0:
            print(f"    No files found for {class_name}!")
            continue
            
        # Calculate how many augmented samples we need
        original_count = len(files)
        needed_augmented = target_samples - original_count
        
        if needed_augmented <= 0:
            # If we have more than enough, just use a subset
            selected_files = random.sample(files, target_samples)
            augmented_files.extend(selected_files)
            augmented_labels.extend([class_name] * target_samples)
            print(f"   Using {target_samples} samples (already sufficient)")
            continue
        
        # Use all original files
        augmented_files.extend(files)
        augmented_labels.extend([class_name] * original_count)
        
        # Calculate how many augmentations per original file
        augmentations_per_file = max(1, needed_augmented // original_count)
        extra_needed = needed_augmented % original_count
        
        print(f"  Creating {augmentations_per_file} augmentations per file")
        
        augmented_count = 0
        # Create augmented versions for each original file
        for file_idx, original_file in enumerate(files):
            try:
                # Load original audio
                audio, sr = librosa.load(original_file, sr=TARGET_SR, mono=True)
                
                # Generate augmented versions
                augmented_versions = aggressive_augmentation(audio, sr, augmentations_per_file)
                
                # Store in cache and add to file list
                for aug_idx, aug_audio in enumerate(augmented_versions):
                    if augmented_count >= needed_augmented:
                        break
                    
                    cache_key = f"aug_{class_name}_{file_idx}_{aug_idx}"
                    AUGMENTED_AUDIO_CACHE[cache_key] = (aug_audio, sr)
                    augmented_files.append(cache_key)
                    augmented_labels.append(class_name)
                    augmented_count += 1
                    
            except Exception as e:
                print(f"  Error augmenting {original_file}: {e}")
                continue
        
        # If we still need more, duplicate some augmented files
        if augmented_count < needed_augmented:
            remaining = needed_augmented - augmented_count
            # Get recently added augmented files
            recent_augmented = [f for f in augmented_files if f.startswith(f"aug_{class_name}")]
            if recent_augmented:
                additional = random.choices(recent_augmented, k=remaining)
                augmented_files.extend(additional)
                augmented_labels.extend([class_name] * remaining)
                augmented_count += remaining
        
        print(f"   Added {augmented_count} augmented samples")
        print(f"   Final: {original_count} + {augmented_count} = {original_count + augmented_count}")
    
    # Verify final counts
    final_counts = Counter(augmented_labels)
    print(f"\n FINAL CLASS DISTRIBUTION:")
    for class_name in class_files.keys():
        count = final_counts.get(class_name, 0)
        print(f"  {class_name}: {count} samples")
    
    return augmented_files, augmented_labels

# --- BIAS CORRECTION FUNCTIONS ---
def apply_hungry_sleepy_bias_correction(y_prob, class_labels):
    hungry_idx = class_labels.index('hungry') if 'hungry' in class_labels else -1
    sleepy_idx = class_labels.index('sleepy') if 'sleepy' in class_labels else -1
    discomfort_idx = class_labels.index('discomfort_pain') if 'discomfort_pain' in class_labels else -1
    
    if hungry_idx == -1 or sleepy_idx == -1:
        return y_prob
    
    corrected_prob = y_prob.copy()
    hungry_prob = y_prob[hungry_idx]
    sleepy_prob = y_prob[sleepy_idx]
    
    if hungry_prob > 0.80 and sleepy_prob > 0.15:
        correction_factor = min(0.15, sleepy_prob * 0.8)
        corrected_prob[hungry_idx] -= correction_factor
        corrected_prob[sleepy_idx] += correction_factor
        corrected_prob = corrected_prob / np.sum(corrected_prob)
    
    discomfort_prob = y_prob[discomfort_idx] if discomfort_idx != -1 else 0
    if (hungry_prob > 0.65 and sleepy_prob > 0.25 and 
        discomfort_prob < 0.1 and sleepy_prob > hungry_prob * 0.4):
        balance_factor = 0.10
        corrected_prob[hungry_idx] -= balance_factor
        corrected_prob[sleepy_idx] += balance_factor
        corrected_prob = corrected_prob / np.sum(corrected_prob)
    
    return corrected_prob

def evaluate_with_bias_correction(classifier, X_test, Y_test, class_labels):
    y_pred_standard = classifier.predict(X_test)
    y_prob_standard = classifier.predict_proba(X_test)
    
    y_pred_corrected = []
    y_prob_corrected = []
    
    for prob_vector in y_prob_standard:
        corrected_prob = apply_hungry_sleepy_bias_correction(prob_vector, class_labels)
        y_prob_corrected.append(corrected_prob)
        y_pred_corrected.append(np.argmax(corrected_prob))
    
    y_pred_corrected = np.array(y_pred_corrected)
    y_prob_corrected = np.array(y_prob_corrected)
    
    print("\n" + "="*60)
    print(" BIAS CORRECTION EVALUATION")
    print("="*60)
    
    standard_accuracy = accuracy_score(Y_test, y_pred_standard)
    corrected_accuracy = accuracy_score(Y_test, y_pred_corrected)
    
    print(f"Standard Accuracy:  {standard_accuracy:.4f}")
    print(f"Corrected Accuracy: {corrected_accuracy:.4f}")
    
    hungry_idx = class_labels.index('hungry')
    sleepy_idx = class_labels.index('sleepy')
    
    hungry_as_sleepy_standard = np.sum((Y_test == hungry_idx) & (y_pred_standard == sleepy_idx))
    sleepy_as_hungry_standard = np.sum((Y_test == sleepy_idx) & (y_pred_standard == hungry_idx))
    
    hungry_as_sleepy_corrected = np.sum((Y_test == hungry_idx) & (y_pred_corrected == sleepy_idx))
    sleepy_as_hungry_corrected = np.sum((Y_test == sleepy_idx) & (y_pred_corrected == hungry_idx))
    
    print(f"\n Hungry-Sleepy Confusion:")
    print(f"Standard - Hungry → Sleepy: {hungry_as_sleepy_standard}")
    print(f"Standard - Sleepy → Hungry: {sleepy_as_hungry_standard}")
    print(f"Corrected - Hungry → Sleepy: {hungry_as_sleepy_corrected}")
    print(f"Corrected - Sleepy → Hungry: {sleepy_as_hungry_corrected}")
    
    improvement = corrected_accuracy - standard_accuracy
    if improvement > 0:
        print(f" Accuracy Improvement: +{improvement:.4f}")
    elif improvement < 0:
        print(f"  Accuracy Decrease: {improvement:.4f}")
    else:
        print(f"  No change in accuracy")
    
    return y_pred_corrected, y_prob_corrected, y_pred_standard, y_prob_standard

# --- VISUALIZATION FUNCTIONS ---
def create_metrics_visualization(y_true, y_pred, y_prob, class_labels, save_path='training_metrics.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Baby Cry Classification Model Performance Metrics', fontsize=16, fontweight='bold')
    
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0,0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    axes[0,0].set_xticks(np.arange(len(class_labels)))
    axes[0,0].set_yticks(np.arange(len(class_labels)))
    axes[0,0].set_xticklabels(class_labels)
    axes[0,0].set_yticklabels(class_labels)
    
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            axes[0,0].text(j, i, str(cm[i, j]),
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black")
    
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    
    x = np.arange(len(class_labels))
    width = 0.25
    
    axes[0,1].bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
    axes[0,1].bar(x, recall, width, label='Recall', alpha=0.8, color='lightcoral')
    axes[0,1].bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightgreen')
    axes[0,1].set_xlabel('Classes')
    axes[0,1].set_ylabel('Scores')
    axes[0,1].set_title('Per-Class Metrics')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(class_labels, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)
    
    overall_metrics = {
        'Accuracy': accuracy,
        'Precision': precision_score(y_true, y_pred, average='macro'),
        'Recall': recall_score(y_true, y_pred, average='macro'),
        'F1-Score': f1_score(y_true, y_pred, average='macro')
    }
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = axes[1,0].bar(overall_metrics.keys(), overall_metrics.values(), color=colors)
    axes[1,0].set_title('Overall Metrics')
    axes[1,0].set_ylabel('Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(0, 1)
    
    for bar, (metric, value) in zip(bars, overall_metrics.items()):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    class_counts = Counter(y_true)
    colors = ['gold', 'lightcoral', 'lightgreen']
    wedges, texts, autotexts = axes[1,1].pie(class_counts.values(), labels=class_counts.keys(), 
                                            autopct='%1.1f%%', startangle=90, colors=colors[:len(class_counts)])
    axes[1,1].set_title('Test Set Class Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Metrics visualization saved to: {save_path}")
    return overall_metrics

# --- MAIN TRAINING CODE ---
def main():
    print("\n" + "="*80)
    print(" 3-CLASS BABY CRY CLASSIFICATION WITH EQUAL AUGMENTATION")
    print("="*80)
    
    # Clear global cache
    global AUGMENTED_AUDIO_CACHE
    AUGMENTED_AUDIO_CACHE = {}
    
    # --- COLLECT AND BALANCE DATASET ---
    print("\n--- COLLECTING AND BALANCING DATASET ---")
    
    # Collect all file paths by class
    class_files = {
        'hungry': [],
        'discomfort_pain': [], 
        'sleepy': []
    }
    
    # Search in class directories
    for class_name in class_files.keys():
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.exists(class_dir):
            for ext in ['*.wav', '*.mp3', '*.caf', '*.m4a']:
                search_path = os.path.join(class_dir, ext)
                file_list = glob.glob(search_path)
                class_files[class_name].extend(file_list)
                print(f"Found {len(file_list)} files in {class_name} directory")
    
    # Also search recursively for any audio files
    for ext in ['*.wav', '*.mp3', '*.caf', '*.m4a']:
        search_path = os.path.join(DATA_DIR, '**', ext)
        file_list = glob.glob(search_path, recursive=True)
        for file_path in file_list:
            label = extract_label_from_filename(file_path)
            if label and label in class_files:
                if file_path not in class_files[label]:
                    class_files[label].append(file_path)
    
    # Print initial distribution
    print("\n INITIAL CLASS DISTRIBUTION:")
    total_files = 0
    for class_name, files in class_files.items():
        print(f"  {class_name}: {len(files)}")
        total_files += len(files)
    print(f"  Total: {total_files} files")
    
    # --- EQUAL AUGMENTATION FOR ALL CLASSES ---
    balanced_files, balanced_labels = augment_class_files(class_files, TARGET_SAMPLES_PER_CLASS)
    
    # Convert to arrays
    Y_raw = np.array(balanced_labels)
    
    # --- FEATURE EXTRACTION ---
    print("\n--- EXTRACTING FEATURES FROM BALANCED DATASET ---")
    
    # Split the balanced dataset
    train_files, test_files, Y_train_raw, Y_test_raw = train_test_split(
        balanced_files, Y_raw, test_size=0.2, random_state=42, stratify=Y_raw
    )
    
    print(f"Training Files: {len(train_files)}, Test Files: {len(test_files)}")
    
    # Extract test features (NO AUGMENTATION)
    X_test = []
    Y_test = []
    
    for file_path in tqdm(test_files, desc="Extracting Test Features"):
        feature = process_audio_and_extract_features_enhanced(file_path, apply_augment=False)
        if feature is not None:
            X_test.append(feature)
            Y_test.append(Y_test_raw[list(test_files).index(file_path)])
    
    # Extract training features (WITH AUGMENTATION)
    X_train = []
    Y_train = []
    
    print("Extracting training features with augmentation...")
    for file_path in tqdm(train_files, desc="Extracting Train Features"):
        # For augmented files, don't apply additional augmentation
        apply_aug = not file_path.startswith("aug_")
        feature = process_audio_and_extract_features_enhanced(file_path, apply_augment=apply_aug)
        if feature is not None:
            X_train.append(feature)
            Y_train.append(Y_train_raw[list(train_files).index(file_path)])
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    print(f"\nFinal feature extraction complete:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # --- SVM TRAINING ---
    print("\n--- SVM CLASSIFIER TRAINING ---")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare labels
    class_labels = sorted(list(set(Y_raw)))
    label_to_index = {label: i for i, label in enumerate(class_labels)}
    index_to_label = {i: label for label, i in label_to_index.items()}
    
    Y_train_indices = np.array([label_to_index[y] for y in Y_train])
    Y_test_indices = np.array([label_to_index[y] for y in Y_test])
    
    print(f"Class labels: {class_labels}")
    print("Final training set distribution:")
    train_counts = Counter(Y_train)
    for class_name in class_labels:
        count = train_counts.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    # --- SVM TRAINING ---
    print("\n Training SVM Classifier...")
    
    svm_classifier = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    svm_classifier.fit(X_train_scaled, Y_train_indices)
    
    # --- EVALUATION WITH BIAS CORRECTION ---
    print("\n--- COMPREHENSIVE MODEL EVALUATION ---")
    
    Y_pred, Y_prob, Y_pred_standard, Y_prob_standard = evaluate_with_bias_correction(
        svm_classifier, X_test_scaled, Y_test_indices, class_labels
    )
    
    accuracy = accuracy_score(Y_test_indices, Y_pred)
    
    print(f"\n Final SVM Training Complete!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(Y_test_indices, Y_pred, target_names=class_labels))
    
    # Print per-class accuracy
    print("\n--- PER-CLASS ACCURACY ---")
    for i, class_name in enumerate(class_labels):
        class_mask = Y_test_indices == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(Y_pred[class_mask] == Y_test_indices[class_mask])
            print(f"  {class_name}: {class_accuracy:.3f} ({np.sum(class_mask)} samples)")
    
    # --- CREATE VISUALIZATIONS ---
    print("\n--- CREATING METRICS VISUALIZATIONS ---")
    
    overall_metrics = create_metrics_visualization(
        Y_test_indices, Y_pred, Y_prob, class_labels, 
        'comprehensive_metrics_3class.png'
    )
    
    # --- SAVE MODEL AND ASSETS ---
    print("\n Saving model and assets...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    joblib.dump(svm_classifier, f'svm_cry_classifier_3class_{timestamp}.pkl')
    joblib.dump(scaler, f'scaler_3class_{timestamp}.pkl')
    
    with open('labels.txt', 'w') as f:
        f.write('\n'.join(class_labels))
    
    model_metadata = {
        'feature_dim': FEATURE_DIM,
        'class_labels': class_labels,
        'label_to_index': label_to_index,
        'whisper_model': MODEL_NAME,
        'target_samples_per_class': TARGET_SAMPLES_PER_CLASS,
        'test_accuracy': float(accuracy),
        'overall_metrics': overall_metrics,
        'training_timestamp': timestamp,
        'num_classes': 3,
        'classes': class_labels,
        'bias_correction_applied': True,
        'initial_data_distribution': {k: len(v) for k, v in class_files.items()}
    }
    
    with open(f'model_metadata_3class_{timestamp}.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"\n 3-CLASS SVM MODEL TRAINING COMPLETE!")
    print(" Generated Visualizations:")
    print("  - comprehensive_metrics_3class.png")
    print("\n Saved Model Files:")
    print(f"  - svm_cry_classifier_3class_{timestamp}.pkl")
    print(f"  - scaler_3class_{timestamp}.pkl")
    print(f"  - model_metadata_3class_{timestamp}.json")
    print(f"  - labels.txt")
    
    print(f"\n Final Performance Summary:")
    print(f"  Accuracy: {overall_metrics['Accuracy']:.3f}")
    print(f"  Precision: {overall_metrics['Precision']:.3f}")
    print(f"  Recall: {overall_metrics['Recall']:.3f}")
    print(f"  F1-Score: {overall_metrics['F1-Score']:.3f}")
    
    print(f"\n Classes: {class_labels}")
    print(f" Equal Augmentation: Applied to all classes")
    print(f" Final class distribution: {TARGET_SAMPLES_PER_CLASS} samples each")

if __name__ == "__main__":
    main()