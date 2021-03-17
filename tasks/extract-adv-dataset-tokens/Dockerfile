FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install torch==1.4.0 numpy dill tqdm torchtext==0.5.0 tensorboard matplotlib pandas

COPY models/pytorch-seq2seq /model

COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
