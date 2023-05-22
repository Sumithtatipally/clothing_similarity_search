# Use the official Python image as the base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask code and data files to the working directory
COPY myntra_products_catalog.csv .
COPY embeddings.npy .

# Copy NLTK data files
COPY nltk_data/ /usr/share/nltk_data/

# Copy the Flask app code to the working directory
COPY opt_test.py .

# Expose the port on which the Flask app will run
EXPOSE 5500

# Start the Flask app
CMD ["python", "opt_test.py"]
