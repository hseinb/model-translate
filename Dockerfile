FROM python:3.8-slim

# Create the working directory
RUN set -ex && mkdir /translate
WORKDIR /translate

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the relevant directories
COPY templates/ ./templates
COPY . ./

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /translate
CMD python3 /translate/app.py
