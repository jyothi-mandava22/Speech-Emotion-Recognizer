from huggingface_hub import HfApi
import time

api = HfApi()

api.create_repo(
    "speech-emotion-recognizer",
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True
)
print("Space created successfully")
print("Waiting for Space to initialize...")
time.sleep(10)

files = [
    ("app/gradio_app.py", "app.py"),
    ("models/ser_model.keras", "models/ser_model.keras"),
    ("src/extract_features.py", "src/extract_features.py"),
    ("src/dataset.py", "src/dataset.py"),
    ("requirements.txt", "requirements.txt"),
]

for local_path, repo_path in files:
    print(f"Uploading {local_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id="jyothi22/speech-emotion-recognizer",
        repo_type="space"
    )
    print(f"Done: {repo_path}")

print("\nAll files uploaded!")
print("Visit: https://huggingface.co/spaces/jyothi22/speech-emotion-recognizer")