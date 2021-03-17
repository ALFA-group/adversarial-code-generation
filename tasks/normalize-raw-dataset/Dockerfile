FROM python:3.7.3

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

COPY vendor/CodeSearchNet/function_parser/ /src/function-parser/

RUN cd /src/function-parser/script && python setup.py
WORKDIR /src/function-parser

COPY task/entrypoint.sh /src/entrypoint.sh

ENTRYPOINT [ "/src/entrypoint.sh" ]
