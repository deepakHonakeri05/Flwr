
FROM tensorflow/tensorflow:latest


RUN pip3 install flwr

RUN mkdir quest
WORKDIR /quest/

COPY ./client1.py /quest/client1.py
