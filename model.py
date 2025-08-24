import os
import librosa
import numpy as np
import joblib
import pandas as pd

# Paths
DATASET_DIR = "C:/Users/Welcome/Desktop/echosentinal/Dataset"  # your dataset root folder
OUTPUT_CSV = "audio_features.csv"

# Classes
class_map = {
    "Marine Animals": 0,
    "Natural Sounds": 1,
    "Human made Objects": 2
}

# Feature extractor
def extract_features(file_path, sr=16000, n_mfcc=20):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)   # mean over time
    mfcc_var = np.var(mfcc, axis=1)     # variance over time
    feature_vector = np.hstack([mfcc_mean, mfcc_var])  # concat
    return feature_vector

# Loop over dataset
rows = []
for label_name, label_id in class_map.items():
    folder = os.path.join(DATASET_DIR, label_name)
    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            fpath = os.path.join(folder, fname)
            features = extract_features(fpath)
            row = features.tolist()
            row.append(label_id)  # add label at end
            rows.append(row)

# Save to CSV
columns = [f"mfcc_mean_{i}" for i in range(20)] + [f"mfcc_var_{i}" for i in range(20)] + ["label"]
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved features to {OUTPUT_CSV}, shape: {df.shape}")
joblib.dump(clf, "rf_model.pkl")
print("💾 Model saved as rf_model.pkl")
