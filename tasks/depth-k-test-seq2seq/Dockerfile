FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install torch numpy dill tqdm torchtext tensorboard matplotlib

COPY models/pytorch-seq2seq /model

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
