import os
from pydub import AudioSegment

# Paths to your data
mp3_dir = r"C:\Users\alexx\OneDrive\Escriptori\uni\4th year\Advanced Machine Learning\project\cv-corpus-18.0-delta-2024-06-14-ca\cv-corpus-18.0-delta-2024-06-14\ca\clips"
wav_dir = r"C:\Users\alexx\OneDrive\Escriptori\uni\4th year\Advanced Machine Learning\project\cv-corpus-18.0-delta-2024-06-14-ca\cv-corpus-18.0-delta-2024-06-14\ca\wav"

# Iterate over all mp3 files and convert them to wav
for file_name in os.listdir(mp3_dir):
    if file_name.endswith(".mp3"):
        mp3_path = os.path.join(mp3_dir, file_name)
        wav_path = os.path.join(wav_dir, os.path.splitext(file_name)[0] + ".wav")

        # Load mp3 file
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Export as wav file with the desired sample rate (16kHz)
        audio.export(wav_path, format="wav", parameters=["-ar", "16000"])  # "-ar" specifies sample rate
        
print("Conversion completed!")