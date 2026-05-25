import os
import glob
import pandas as pd

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def build_dataframe(data_dir: str) -> pd.DataFrame:
    records = []
    for wav_path in glob.glob(f"{data_dir}/**/*.wav", recursive=True):
        fname = os.path.basename(wav_path).replace(".wav", "")
        parts = fname.split("-")
        if len(parts) != 7:
            continue
        emotion_code = parts[2]
        records.append({
            "path": wav_path,
            "emotion": EMOTION_MAP.get(emotion_code, "unknown"),
            "emotion_id": int(emotion_code) - 1,
            "actor": int(parts[6])
        })
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = build_dataframe("data/raw")
    print(f"Total files: {len(df)}")
    print(df["emotion"].value_counts())
    print(df.head())