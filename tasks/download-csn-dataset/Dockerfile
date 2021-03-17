FROM debian:stretch

RUN apt-get update && apt-get install -y --no-install-recommends parallel jq unzip curl ca-certificates

COPY task/entrypoint.sh /

ENTRYPOINT [ "/entrypoint.sh" ]
