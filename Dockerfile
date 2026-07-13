FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# ✅ Usar formato shell para que $PORT sea expandido
CMD gunicorn app:app --bind 0.0.0.0:$PORT