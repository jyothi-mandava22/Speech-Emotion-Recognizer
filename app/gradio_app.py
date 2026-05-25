import gradio as gr
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from extract_features import extract_features

model = tf.keras.models.load_model("models/ser_model.keras")

EMOTIONS = ["neutral", "calm", "happy", "sad",
            "angry", "fearful", "disgust", "surprised"]

EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲"
}

def predict_emotion(audio_path):
    if audio_path is None:
        return "No audio received. Please record something."

    features = extract_features(audio_path)
    features = features[np.newaxis, ..., np.newaxis]
    probs = model.predict(features, verbose=0)[0]

    results = {
        f"{EMOJI[EMOTIONS[i]]} {EMOTIONS[i]}": float(probs[i])
        for i in range(len(EMOTIONS))
    }
    return results

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload speech (3 seconds)"),
    outputs=gr.Label(num_top_classes=4, label="Detected Emotion"),
    title="🎙️ Speech Emotion Recognizer",
    description="Record 3 seconds of speech and the model will detect the emotion. Trained on RAVDESS dataset with 8 emotion classes.",
    examples=None
)

if __name__ == "__main__":
    demo.launch(share=True)