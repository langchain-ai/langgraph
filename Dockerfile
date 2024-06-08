FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --with test

RUN poetry run pytest
