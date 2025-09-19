# Use a base image matching the specified PyTorch and CUDA version
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the serverless handler script
COPY runpod_handler.py .

# Set the command to start the RunPod worker
CMD ["python", "runpod_handler.py"]
