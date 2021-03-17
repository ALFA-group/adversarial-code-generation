sudo mv ./datasets/adversarial/renamevar-param/tokens/sri/py150/ss* ./final-results/plots/$1

sudo /home/shash/.pyenv/shims/python ./models/pytorch-seq2seq/seq2seq/util/plots.py ./final-results/plots/$1
