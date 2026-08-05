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

# Run the app via Uvicorn.
# 2 workers: each gets its own copy of weaviate_client/gemini_client (fine — they're
# stateless HTTP clients). t3.medium has 4 GB so 2 is the safe ceiling.
# uvloop + httptools are drop-in C-level replacements that improve async throughput ~40%.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "2", "--loop", "uvloop", "--http", "httptools"]
