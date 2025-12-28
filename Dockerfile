FROM python:3.11-slim

WORKDIR /app

# Copy all backend files
COPY backend/requirements.txt ./requirements.txt
COPY backend/train_model.py ./train_model.py
COPY backend/recommendations.py ./recommendations.py
COPY backend/app ./app

# Create models directory and copy model file
RUN mkdir -p ./models
COPY backend/models/abandonment_model.joblib ./models/abandonment_model.joblib

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verify model exists
RUN ls -la ./models/

# Expose port
EXPOSE 8000

# Start command
CMD ["sh", "-c", "cd app && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
