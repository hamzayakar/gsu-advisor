# Use official lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Gradio will run on
EXPOSE 7860

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "app.py"]