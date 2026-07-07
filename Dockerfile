FROM python:3.11-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    iputils-ping curl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /flcore

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ln -s /usr/bin/python3 /usr/bin/python
