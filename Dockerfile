# Multi-stage Dockerfile for RAG Chatbot

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create wheels directory and install packages
RUN mkdir /wheels && \
    pip install --wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install python packages from wheels
RUN pip install --no-index --find-links /wheels -r requirements.txt && \
    rm -rf /wheels

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs vectorstore data evaluation

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu
ENV PORT=8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import streamlit; print('healthy')" || exit 1

# Expose port for Streamlit
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
