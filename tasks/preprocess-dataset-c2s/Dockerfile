FROM python:3.7-stretch

RUN pip3 install numpy tensorflow tqdm joblib sklearn fissix

RUN apt-get update && apt-get install -y openjdk-8-jre

COPY task/entrypoint.sh /entrypoint.sh
COPY task/pystage.py /pystage.py

COPY vendor/code2seq /code2seq

ENTRYPOINT [ "/entrypoint.sh" ]
