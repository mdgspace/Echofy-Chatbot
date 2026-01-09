# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container
COPY requirements.txt .

# Install CPU-only Torch FIRST
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code to the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Make the Entrypoint executable
RUN chmod +x entrypoint.sh

# Execute the Entrypoint script
ENTRYPOINT ["./entrypoint.sh"]