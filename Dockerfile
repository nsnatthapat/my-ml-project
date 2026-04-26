FROM python:3.9-slim

WORKDIR /app

# System libs required by LightGBM (OpenMP) and scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the project package
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Copy application code and artifacts
COPY app/ app/
COPY data/ data/
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
