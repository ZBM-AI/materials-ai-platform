FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY materials_ai/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY materials_ai/ /app/materials_ai/

ENV PYTHONPATH=/app

EXPOSE 8501
CMD ["streamlit", "run", "materials_ai/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
