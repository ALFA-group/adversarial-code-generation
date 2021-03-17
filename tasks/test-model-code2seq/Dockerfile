FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get install -y nvidia-modprobe

RUN pip3 install requests rouge tqdm

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY models/code2seq-merged /code2seq
COPY models/pytorch-seq2seq /seq2seq
COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
