FROM python:3.11-slim
WORKDIR /app

# Copy backend directory contents
COPY requirements.txt ./
COPY train_model.py ./
COPY recommendations.py ./
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["sh", "-c", "cd app && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
