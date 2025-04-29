FROM ubuntu:22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --assume-yes \
    pip wkhtmltopdf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /home/requirements.txt
RUN pip3 install -r /home/requirements.txt --no-cache-dir
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /flcore
