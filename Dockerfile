# Use Python 3.12 for compatibility with the research dependencies.
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies.
COPY backend/requirements.txt ./backend/requirements.txt
RUN python -m pip install --no-cache-dir -r backend/requirements.txt

# Copy app sources and frontend assets.
COPY backend ./backend
COPY frontend ./frontend

WORKDIR /app/backend

EXPOSE 5000

CMD ["python", "app.py"]
