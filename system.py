# Paths
MODEL_PATH = "rf_model.pkl"  # trained RandomForest saved with joblib
TEST_FILE = "test_mix_5.wav"  # audio to analyze

# Classes
class_map = {
    0: "Marine Animals",
    1: "Natural Sounds",
    2: "Human made Objects"
}

# Feature extractor
def extract_features(y, sr=16000, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)
    feature_vector = np.hstack([mfcc_mean, mfcc_var])
    return feature_vector

# Segment into fixed windows and classify
def segment_and_classify(file_path, model, window_size=1.0):
    y, sr = librosa.load(file_path, sr=16000)  
    total_duration = librosa.get_duration(y=y, sr=sr)

    results = []
    classes_detected = set()

    for start in np.arange(0, total_duration, window_size):
        end = min(start + window_size, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        segment = y[start_sample:end_sample]

        if len(segment) < sr * 0.2:  # skip very tiny segments (<0.2s)
            continue

        feat = extract_features(segment, sr).reshape(1, -1)
        pred = model.predict(feat)[0]

        results.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "label": class_map[pred]
        })
        classes_detected.add(class_map[pred])

    return results, classes_detected

# Merge consecutive segments with same label
def merge_segments(segments):
    if not segments:
        return []

    merged = [segments[0]]

    for seg in segments[1:]:
        if seg["label"] == merged[-1]["label"]:
            # Extend the last merged segment
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


# Example run
if __name__ == "__main__":
    # Load trained RF model
    model = joblib.load(MODEL_PATH)

    segments, classes_detected = segment_and_classify(TEST_FILE, model)

    # Merge consecutive duplicates
    merged_segments = merge_segments(segments)

    print("\nDetected segments (merged):")
    for seg in merged_segments:
        print(f"[{seg['start']}s - {seg['end']}s] → {seg['label']}")

    print("\n Classes detected in this file:", list(classes_detected))
