FROM ubuntu:22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --assume-yes \
    pip iputils-ping curl wget wkhtmltopdf

COPY requirements.txt /home/requirements.txt
RUN pip3 install -r /home/requirements.txt
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /flcore
COPY . /flcore