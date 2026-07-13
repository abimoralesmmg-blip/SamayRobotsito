FROM python:3.11-slim-bookworm

# Instalar dependencias del sistema para OpenCV y otras librerías
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements.txt primero (para aprovechar caché)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto (Railway asignará uno automáticamente)
EXPOSE 8080

# Comando para ejecutar la aplicación con Gunicorn
# - workers=1 para ahorrar memoria (solo un proceso)
# - threads=2 para manejar concurrencia con hilos
# - timeout=120 para evitar que se cuelgue en predicciones largas
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120