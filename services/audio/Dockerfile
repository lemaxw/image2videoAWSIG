FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*

COPY services/audio/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/audio/audio_service.py /app/audio_service.py

EXPOSE 8000
CMD ["uvicorn", "audio_service:app", "--host", "0.0.0.0", "--port", "8000"]
