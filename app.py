from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from defect_cnn_pytorch import predict_image


app = FastAPI(title="Defect Detection API")


# Locate the frontend folder.
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


# Make CSS and JavaScript files available.
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)


# First page: About Project
@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "about.html")


# Model prediction page
@app.get("/model")
def model_page():
    return FileResponse(FRONTEND_DIR / "index.html")


# Receive an uploaded image and predict its class.
@app.post("/predict")
def predict(file: UploadFile = File(...)):

    allowed_types = ["image/png", "image/jpeg"]

    if file.content_type not in allowed_types:
        return {
            "error": "Please upload a PNG or JPEG image"
        }

    file_extension = Path(file.filename or "").suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension,
    ) as temporary_file:

        shutil.copyfileobj(file.file, temporary_file)
        temporary_path = Path(temporary_file.name)

    try:
        result = predict_image(temporary_path)
        return result

    finally:
        temporary_path.unlink(missing_ok=True)