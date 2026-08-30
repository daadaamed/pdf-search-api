FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch explicitly first. Satisfying torch here first means
# the requirements.txt install below won't pull the GPU build in afterward.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]