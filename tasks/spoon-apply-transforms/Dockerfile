FROM python:3.7.3-stretch

# FROM https://github.com/github/CodeSearchNet/blob/b1bf7b06908762011563106026c97c75c26d5bd1/function_parser/Dockerfile
RUN touch /etc/inside-container

RUN set -ex && pip3 install --upgrade pip
RUN set -ex && pip3 --no-cache-dir install --upgrade jupyter \
    tree_sitter \
    requests \
    pyhive \
    tqdm \
    pandas \
    python-arango \
    docopt \
    elasticsearch \
    dpu_utils

RUN mkdir -p /src/vendor
RUN mkdir -p /src/build
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-python.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-javascript.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-typescript.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-go.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-ruby.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-java.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-cpp.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-c-sharp.git
RUN cd /src/vendor && git clone https://github.com/tree-sitter/tree-sitter-php.git

RUN apt-get update && apt-get install --no-install-recommends -y jq parallel

RUN set -ex && pip3 --no-cache-dir install --upgrade sentencepiece javalang fissix

RUN curl -Lo /openjdk.tar.gz https://github.com/AdoptOpenJDK/openjdk9-binaries/releases/download/jdk-9.0.4%2B11/OpenJDK9U-jdk_x64_linux_hotspot_9.0.4_11.tar.gz && \
  tar -xf /openjdk.tar.gz && \
  export PATH=/jdk-9.0.4+11/bin:$PATH

COPY vendor/CodeSearchNet/function_parser/ /src/function-parser/

RUN cd /src/function-parser/script && python setup.py

WORKDIR /app

RUN mkdir /mnt/raw-outputs

COPY task/jars/ /app/

COPY task/ /app/

RUN \
  /jdk-9.0.4+11/bin/javac -cp /app/spoon.jar:/app/gson.jar:/app transforms/*.java && \
  /jdk-9.0.4+11/bin/javac -cp /app/spoon.jar:/app/gson.jar:/app Transforms.java

ENTRYPOINT [ "/app/entrypoint.sh" ]
