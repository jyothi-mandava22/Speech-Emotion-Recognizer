from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="app/gradio_app.py",
    path_in_repo="app.py",
    repo_id="jyothi22/speech-emotion-recognizer",
    repo_type="space"
)
print("Done")