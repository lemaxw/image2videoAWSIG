FROM python:3.11-slim

WORKDIR /app

COPY services/memory/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/memory /app/services/memory

ENV PYTHONPATH=/app
ENV MEMPALACE_DISABLE_STDIO_REDIRECT=1
CMD ["uvicorn", "services.memory.memory_service:app", "--host", "0.0.0.0", "--port", "8090"]
