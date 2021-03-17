FROM debian:stretch

RUN apt-get update && apt-get install -y --no-install-recommends parallel jq unzip wget ca-certificates

COPY task/ /

ENTRYPOINT [ "/entrypoint.sh" ]
