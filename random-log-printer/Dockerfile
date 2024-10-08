# Use a base Python image
FROM python:3.9-slim

# Install necessary packages
RUN apt-get update && apt-get install -y git

# Set the working directory
WORKDIR /app

# Copy the Python script that will run the logic
COPY main.py /app/


# Run the Python script
CMD ["python", "main.py"]