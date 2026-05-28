FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn torch transformers
COPY ../../materials_ai /app/materials_ai
COPY app.py /app/

ENV PYTHONPATH=/app
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
