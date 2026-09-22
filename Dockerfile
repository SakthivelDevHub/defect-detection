# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY app.py .
# COPY defect_cnn_pytorch.py .
# COPY defect_cnn_best.pth .
# COPY frontend ./frontend

# EXPOSE 8000

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch without CUDA libraries.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && pip install --no-cache-dir \
        -r requirements.txt

COPY app.py .
COPY defect_cnn_pytorch.py .
COPY defect_cnn_best.pth .
COPY frontend ./frontend

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]