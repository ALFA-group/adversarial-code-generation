FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get install -y nvidia-modprobe

RUN pip3 install requests rouge tqdm pandas

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY models/code2seq-merged /model

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
