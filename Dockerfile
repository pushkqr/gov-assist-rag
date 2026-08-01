FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure persistent directories and files exist
RUN mkdir -p scratch temp docs && touch mimir_portal.db

# Expose FastAPI port
EXPOSE 8000

# Run the app via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
