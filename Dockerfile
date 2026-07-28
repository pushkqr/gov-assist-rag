FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure scratch, temp, and local_qdrant_db directories exist
RUN mkdir -p scratch temp local_qdrant_db docs

# Expose FastAPI port
EXPOSE 8000

# Run the app via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
