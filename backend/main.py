from fastapi import FastAPI, UploadFile, File, Form
from uuid import uuid4
import os
from pathlib import Path

from train_model import train_model

app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"


@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}


@app.post("/upload-sample")
async def upload_sample(
    class_name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    class_folder = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_folder, exist_ok=True)

    saved_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid4()}{ext}"

        file_path = os.path.join(class_folder, unique_filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        saved_files.append(unique_filename)

    return {
        "message": f"{len(saved_files)} images uploaded successfully",
        "class_name": class_name,
        "files": saved_files
    }


@app.post("/train")
def train():
    result = train_model()

    return {
        "message": "Training completed successfully",
        "details": result
    }