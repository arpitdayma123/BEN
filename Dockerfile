# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg and other dependencies for opencv-python and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY rp_handler.py .
COPY ben_base.py .
COPY requirements.txt .
# config.json is not strictly needed by the handler but copying it just in case any future model loading changes require it.
COPY config.json . 

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container (if needed, though RunPod handles port mapping)
# EXPOSE 80 
# For RunPod serverless, EXPOSE is usually not necessary as it's handled by their infrastructure.

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run rp_handler.py when the container launches
CMD ["python3", "-u", "rp_handler.py"]
