FROM python:3.7.3-stretch


# Maintainer info
LABEL maintainer="aahancharak@gmail.com"

# Working Directories
RUN  mkdir -p  /phytomonk-classification-api
WORKDIR  /phytomonk-classification-api


# pip upgrade with no cache
RUN pip install --no-cache-dir -U pip

# Copy requirements file
COPY requirements.txt .

# Install the application dependencies
RUN pip install -r requirements.txt

# Copy every file from source folder to created directory
COPY  . .

# Run the python application
CMD ["python", "main.py"]