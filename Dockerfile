# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements file first (for caching efficiency)
COPY requirements.txt .

# Ensure pip and dependencies install cleanly
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
