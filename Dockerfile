# Base image
FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Install system dependencies
RUN apt update -y && apt install -y awscli

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run application
CMD ["python3", "app.py"]