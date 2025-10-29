# remove_problematic_files.py
import os
import glob
import shutil

def clean_discomfort_folder():
    discomfort_folder = 'cry_data/donateacry_corpus_cleaned_and_updated_data/discomfort'
    
    if not os.path.exists(discomfort_folder):
        print(" Discomfort folder not found!")
        return
    
    # Find all -ch.wav files
    ch_files = glob.glob(os.path.join(discomfort_folder, '*-ch.wav'))
    
    if not ch_files:
        print(" No problematic -ch.wav files found!")
        return
    
    # Create backup folder
    backup_folder = 'cry_data/questionable_files'
    os.makedirs(backup_folder, exist_ok=True)
    
    print(f" Moving {len(ch_files)} problematic files:")
    for file_path in ch_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(backup_folder, filename)
        shutil.move(file_path, dest_path)
        print(f"   {filename} -> questionable_files/")
    
    # Count remaining files
    remaining_files = glob.glob(os.path.join(discomfort_folder, '*.wav')) + glob.glob(os.path.join(discomfort_folder, '*.mp3'))
    print(f"\n Cleaned discomfort folder: {len(remaining_files)} files remaining")

if __name__ == "__main__":
    clean_discomfort_folder()