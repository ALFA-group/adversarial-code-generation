FROM python:3.7-stretch

RUN apt-get update && apt-get install -y --no-install-recommends parallel jq zip curl ca-certificates && pip install tqdm

COPY task/ /

ENTRYPOINT [ "/entrypoint.sh" ]
