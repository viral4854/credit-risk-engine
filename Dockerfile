# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API when the container starts
# We use "0.0.0.0" to allow external access outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]