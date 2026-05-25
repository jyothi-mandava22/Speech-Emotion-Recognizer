import librosa
import numpy as np
import os
import tqdm
import pandas as pd

def extract_features(file_path: str,
                     sr: int = 22050,
                     duration: float = 3.0) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=sr, duration=duration)

    # Pad or trim to fixed length
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # MFCCs — captures vocal tract shape
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)        # (40, T)

    # Mel spectrogram — captures frequency energy over time
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)               # (40, T)

    # Chroma — captures pitch class
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)  # (12, T)

    # Stack into (92, T)
    combined = np.vstack([mfcc, mel_db, chroma])
    return combined

def batch_extract(df: pd.DataFrame,
                  cache_path: str = "data/features.npy") -> tuple:
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True).item()
        return data["X"], data["y"]

    X, y = [], []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        feat = extract_features(row["path"])
        X.append(feat)
        y.append(row["emotion_id"])

    X = np.array(X)
    y = np.array(y)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, {"X": X, "y": y})
    print(f"Features saved to {cache_path}")
    return X, y

if __name__ == "__main__":
    from dataset import build_dataframe
    df = build_dataframe("data/raw")
    X, y = batch_extract(df)
    print(f"Feature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")