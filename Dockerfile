FROM python:3.12

WORKDIR /app

# Install system dependencies including Java (required for pyspark/HSFS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME (default-jdk installs to different path)
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy .env file (KTHcloud doesn't support environment variables)
# NOTE: Make sure .env is NOT in .gitignore if you want to include it
# For production, consider using a secrets management system
COPY .env .env

# Set Hopsworks host (for serverless)
ENV HOPSWORKS_HOST=c.app.hopsworks.ai
ENV HOPSWORKS_ENGINE=python

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

