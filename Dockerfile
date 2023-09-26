FROM ubuntu

RUN apt-get update -y && apt-get upgrade -y && apt-get install python3 pip iputils-ping curl wget -y
COPY requirements.txt /home/requirements.txt
RUN pip3 install -r /home/requirements.txt

WORKDIR /flcore
