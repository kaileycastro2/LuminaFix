FROM python:3.12-slim

# Install system libraries needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgles2 \
    libegl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download MediaPipe FaceLandmarker model
RUN mkdir -p /app/models && \
    curl -sSL -o /app/models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Copy the rest of the app
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "gunicorn web.app:app --bind 0.0.0.0:${PORT:-8000} --workers ${WEB_WORKERS:-4} --worker-class uvicorn.workers.UvicornWorker --preload --timeout 120"]
