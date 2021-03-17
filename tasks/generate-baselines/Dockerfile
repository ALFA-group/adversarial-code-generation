FROM python:3.7.3

RUN apt-get update && apt-get install -y --no-install-recommends jq

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]