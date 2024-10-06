
FROM python:3.12


WORKDIR /app


COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt


COPY ./app /app/app


CMD ["fastapi", "run", "app/main.py", "--port", "80"]