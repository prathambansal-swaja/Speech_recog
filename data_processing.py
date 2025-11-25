from common import load_metadata, add_clip_lengths, BASE_DIR ,pad_or_truncate,filter_background_and_short
from tqdm import tqdm
import librosa
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

# Load metadata and clip durations
data = load_metadata()
data = add_clip_lengths(data)
data=filter_background_and_short(data, max_len=1.1)
# Load raw waveforms
raw_data = []
for rel_path in tqdm(data["file_name"]):
    full_path = os.path.join(BASE_DIR, rel_path)   # Construct full path
    signal, rate = librosa.load(full_path, sr=16000)
    raw_data.append(signal)

data["raw_data"] = raw_data
print(data.head())
''' padding - make all file same length(16000) '''
# if len of audio is < 16000 we add zeros.
# if len of audio is > 16000 we truncate.

#pad_seq = pad_or_truncate(data, target_len=16000, column="raw_data")
data, pad_seq = pad_or_truncate(data, target_len=16000, column="raw_data", save_csv=True, csv_name="data_final.csv")

print("Padded shape:", pad_seq.shape)
print(data.head())

spec = librosa.feature.melspectrogram(y = data['pad_seq'].values[0], sr = 16000, n_mels = 64)
spec = librosa.power_to_db(spec, ref=np.max)
spec.shape
plt.figure(figsize=(10,4))
librosa.display.specshow(spec , x_axis='time', y_axis='mel',sr=16000,fmax=8000)
plt.colorbar(format = '%+2.0f dB' )
plt.title('Mel-frequency spectrogram bed/00176480_nohash_0.wav')
plt.tight_layout()
plt.show()