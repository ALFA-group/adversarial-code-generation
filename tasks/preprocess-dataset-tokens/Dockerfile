FROM python:3.7-stretch

RUN pip install tqdm

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
