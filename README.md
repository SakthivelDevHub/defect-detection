# AI-Powered Defect Detection

A full-stack deep-learning application that classifies grayscale product images into four categories: **normal**, **scratch**, **crack**, and **stain**. The application accepts PNG or JPEG uploads through a browser, preprocesses each image to match the model input, and returns the predicted class with a confidence percentage. It uses a custom PyTorch convolutional neural network (CNN), a FastAPI backend, an HTML/CSS/JavaScript frontend, and Docker for reproducible deployment.

## Business Problem

Manual visual inspection on a production line can be slow, inconsistent, and difficult to scale. This project demonstrates how computer vision can assist quality inspection by automatically identifying common surface-defect categories from product images.

This is an educational prototype, not a production-ready inspection system. It should be validated with representative factory data before being used for real manufacturing decisions.

## Defect Classes

The model predicts one of four classes:

| Label | Class | Meaning |
| ---: | --- | --- |
| 0 | `normal` | No visible defect |
| 1 | `scratch` | Elongated surface damage |
| 2 | `crack` | Cracked or broken-looking region |
| 3 | `stain` | Localized discoloured or dark region |

## Application Workflow

1. The user opens the About Project landing page.
2. The user navigates to the model page.
3. The user selects a PNG or JPEG image.
4. The backend converts the image to grayscale and resizes it to `32 x 32` pixels.
5. The trained CNN performs inference.
6. The webpage displays the predicted class and confidence percentage.

## Dataset and Preprocessing

- Total images: **1,200**
- Training images: **840** (70%)
- Validation images: **180** (15%)
- Test images: **180** (15%)
- Test samples per class: **45**
- Image format used by the model: **single-channel grayscale**
- Input tensor shape: **`(batch_size, 1, 32, 32)`**
- Pixel values are normalized before inference.
- Splits are stratified so each class is represented proportionally.

The test set remains separate from the training and validation data. It is used only for final evaluation and deployment-pipeline verification.

## Model

The classifier is a custom convolutional neural network implemented in PyTorch. The trained weights are stored in:

```text
defect_cnn_best.pth
```

Training uses mini-batches and evaluates validation loss and accuracy after each epoch. The saved model is then loaded by the FastAPI application for inference.

## Performance

### Full held-out test set

The model achieved:

```text
Final test accuracy: 90.56%
Test images: 180
```

### Docker API smoke test

Twenty exported samples from the held-out test split were uploaded through the Dockerized `/predict` endpoint.

| Actual class | Correct | Subset accuracy |
| --- | ---: | ---: |
| Normal | 5/5 | 100% |
| Scratch | 2/5 | 40% |
| Crack | 5/5 | 100% |
| Stain | 5/5 | 100% |
| **Overall** | **17/20** | **85%** |

The 20-image result is a small, non-random smoke-test subset and must not replace the full 180-image test accuracy. Its purpose is to confirm that browser upload, preprocessing, API inference, and response rendering work together inside Docker.

The subset also reveals that scratches can be confused with cracks. Three scratch samples were classified as cracks.

## Technology Stack

- **Model:** PyTorch
- **Data processing:** NumPy, Pandas and scikit-learn
- **Image processing:** Pillow
- **Backend/API:** FastAPI and Uvicorn
- **Frontend:** HTML, CSS and JavaScript
- **Containerization:** Docker

## Project Structure

```text
defect-detection/
|-- frontend/
|   |-- about.html
|   |-- index.html
|   |-- script.js
|   `-- style.css
|-- test_samples/
|   `-- exported held-out test images
|-- app.py
|-- defect_cnn_pytorch.py
|-- defect_cnn_best.pth
|-- defect_detection_images.csv
|-- requirements.txt
|-- Dockerfile
|-- .dockerignore
`-- README.md
```

The dataset and local test samples are not required inside the runtime Docker image. Inference uses the saved model weights.

## Running Locally

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the FastAPI application

Run this from the project directory:

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Running with Docker

Docker Desktop must be installed and its engine must be running.

### 1. Build the image

```powershell
docker build -t defect-detection-app .
```

### 2. Create and run the container

```powershell
docker run --name defect-detection-container -p 8000:8000 defect-detection-app
```

Open:

```text
http://localhost:8000
```

`0.0.0.0:8000` is Uvicorn's listening address inside the container. Use `localhost:8000` in the browser because Docker maps the host's port `8000` to the container's port `8000`.

### Reuse the existing container

If the named container already exists, start it instead of running `docker run` again:

```powershell
docker start -a defect-detection-container
```

To stop it:

```powershell
docker stop defect-detection-container
```

## API

### `POST /predict`

Accepts an uploaded PNG or JPEG image as multipart form data.

Example request:

```powershell
curl.exe -X POST -F "file=@test_samples/crack_1.png" http://localhost:8000/predict
```

Example response:

```json
{
  "label_name": "crack",
  "confidence_percentage": 74.21
}
```

The exact response may contain additional fields depending on the current implementation.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | About Project landing page |
| `GET` | `/model` | Image-upload and prediction page |
| `POST` | `/predict` | CNN prediction API |
| `GET` | `/static/...` | Frontend static assets |

## Known Limitations

- The model was trained on small `32 x 32` grayscale images.
- Performance on real factory photographs has not been established.
- Scratch and crack classes can have similar visual features and may be confused.
- Confidence is not the same as correctness or overall model accuracy.
- The current application supports PNG and JPEG uploads only.
- The prototype does not yet include authentication, persistent prediction history, monitoring, or an upload-size limit.
- More diverse real-world data would be needed before production use.

## Future Improvements

- Collect and label more representative factory images.
- Investigate scratch-versus-crack errors with a confusion matrix and misclassified samples.
- Add data augmentation and compare improved CNN architectures.
- Add automated unit, API and integration tests.
- Add upload-size validation and stronger error handling.
- Record model version and inference metadata.
- Add monitoring for accuracy drift after deployment.
- Deploy the Docker image to a cloud hosting service.

## Current Status

- [x] Dataset preprocessing
- [x] CNN training and evaluation
- [x] Saved model inference
- [x] FastAPI prediction endpoint
- [x] Frontend upload interface
- [x] About Project landing page
- [x] Docker image and local container testing
- [x] End-to-end prediction smoke test
- [ ] Git repository and versioned commits
- [ ] GitHub repository
- [ ] Online deployment
