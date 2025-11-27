# Use an official Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Disable Python output buffering
ENV PYTHONUNBUFFERED=1

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the script when the container starts
CMD ["python3", "update_ssi_prices.py", "--mode", "auto"]