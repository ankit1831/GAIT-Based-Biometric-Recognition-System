from fastapi import FastAPI, UploadFile, File
import zipfile
import os
import shutil
from app.utils import load_silhouette_folder
from app.inference import run_inference

app = FastAPI()

UPLOAD_DIR = "uploads/"
EXTRACT_DIR = "extracted/"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Gait Recognition API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    zip_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save ZIP
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract ZIP
    extract_path = os.path.join(EXTRACT_DIR, file.filename.split(".")[0])

    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)

    os.makedirs(extract_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    silhouettes = load_silhouette_folder(extract_path)

    user, confidence = run_inference(silhouettes)

    return {
        "prediction": user,
        "confidence": confidence
    }