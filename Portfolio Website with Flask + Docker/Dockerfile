# Use a lightweight Python base image
FROM python:3.9-alpine

# Set a working directory inside the container
WORKDIR /app

# Install Flask directly
RUN pip install --no-cache-dir Flask

# Copy the rest of the application code
COPY . .

# Expose the default Flask port
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Define the command to run the Flask app
CMD ["python", "app.py"]
