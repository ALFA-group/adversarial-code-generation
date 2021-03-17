FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN pip install torch==1.6.0 numpy==1.19.2 black==19.10b0 tqdm==4.50.0 pytorch-lightning==0.9.0 wandb==0.10.4 dill tensorboard

COPY models/pytorch-code2seq /code2seq
COPY models/pytorch-seq2seq /seq2seq
COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
