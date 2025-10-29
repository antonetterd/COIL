# verify_audio.py
import os
import librosa
import sounddevice as sd
import numpy as np

def listen_and_verify(audio_file, expected_label):
    print(f"\n🎵 Playing: {os.path.basename(audio_file)}")
    print(f"Expected label: {expected_label}")
    
    audio, sr = librosa.load(audio_file, sr=16000)
    sd.play(audio, sr)
    sd.wait()  # Wait until playback is finished
    
    user_label = input("What does this ACTUALLY sound like? (hungry/pain/discomfort/sleepy): ")
    return user_label.strip().lower()

# Test your files
files_to_check = [
    ("cry_data/sample2.wav", "hungry"),
    ("cry_data/sample3.wav", "discomfort"),  # This is the suspicious one
    ("cry_data/sample4.wav", "hungry"),
    ("cry_data/sample1.wav", "pain")
]

print(" LISTEN TO EACH CRY AND VERIFY THE LABELS:")
print("=" * 50)

for file_path, expected in files_to_check:
    if os.path.exists(file_path):
        actual = listen_and_verify(file_path, expected)
        print(f"Result: Expected '{expected}', You heard '{actual}'")
        print("-" * 50)