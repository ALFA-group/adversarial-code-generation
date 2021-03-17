from typing import Dict

import torch
from torch import nn

from configs import EncoderConfig
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_TYPES
import numpy as np

class PathEncoder(nn.Module):
    def __init__(
        self,
        config: EncoderConfig,
        out_size: int,
        n_subtokens: int,
        subtoken_pad_id: int,
        n_types: int,
        type_pad_id: int,
    ):
        super().__init__()
        self.type_pad_id = type_pad_id
        self.num_directions = 2 if config.use_bi_rnn else 1
        self.n_subtokens = n_subtokens

        self.subtoken_embedding = nn.Linear(n_subtokens, config.embedding_size, bias=False)
        shp = self.subtoken_embedding.weight[:, subtoken_pad_id].shape
        self.subtoken_embedding.weight.data[:, subtoken_pad_id] = torch.zeros(shp[0], requires_grad=True)
        self.subtoken_embedding1 = nn.Embedding(n_subtokens, config.embedding_size, padding_idx=subtoken_pad_id)

        self.type_embedding = nn.Embedding(n_types, config.embedding_size, padding_idx=type_pad_id)

        self.dropout_rnn = nn.Dropout(config.rnn_dropout)
        self.path_lstm = nn.LSTM(
            config.embedding_size,
            config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )

        concat_size = config.embedding_size * 2 + config.rnn_size * self.num_directions
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.linear = nn.Linear(concat_size, out_size, bias=False)
        self.norm = nn.LayerNorm(out_size)
        self.embed_weights = None
    
    def copy_weights(self):
        self.embed_weights = self.subtoken_embedding.weight.clone().permute(1,0)
     
    def forward(self, contexts: Dict[str, torch.Tensor], use_embedding_layer: bool=True, already_one_hot: bool=False) -> torch.Tensor:
        # [max name parts; total paths]
        from_token = contexts[FROM_TOKEN]
        to_token = contexts[TO_TOKEN]

        if use_embedding_layer:
            encoded_from_tokens = self.subtoken_embedding1(from_token).sum(0)
            encoded_to_tokens = self.subtoken_embedding1(to_token).sum(0)
        else:
            # to_token1: total paths x max name parts x |V|
            # to_token1 = torch.zeros(to_token1.size(0), to_token1.size(1), self.n_subtokens, device=d).scatter_(2, to_token1.unsqueeze(2), 1.).squeeze()
            if not already_one_hot:
                to_token1 = to_token.permute(1,0)
                to_token1 = torch.nn.functional.one_hot(to_token1, num_classes=self.n_subtokens)
                to_token1 = to_token1.float()
            else:
                to_token1 = to_token.permute(1, 0, 2)
            
            self.copy_weights()
            # total paths x max name parts x 128 -> sum(1) -> total paths x 128
            encoded_to_tokens = torch.matmul(to_token1, self.embed_weights).sum(1)

            # from_token1: total paths x max name parts x |V|
            #from_token1 = torch.zeros(from_token1.size(0), from_token1.size(1), self.n_subtokens, device=d).scatter_(2, from_token1.unsqueeze(2), 1.).squeeze()
            if not already_one_hot:
                from_token1 = from_token.permute(1, 0)
                from_token1 = torch.nn.functional.one_hot(from_token1, num_classes=self.n_subtokens)
                from_token1 = from_token1.float()
            else:
                from_token1 = from_token.permute(1, 0, 2)
            encoded_from_tokens = torch.matmul(from_token1, self.embed_weights).sum(1)

        # [max path length; total paths]
        path_types = contexts[PATH_TYPES]
        # [max path length; total paths; embedding size]
        path_types_embed = self.type_embedding(path_types)

        # create packed sequence (don't forget to set enforce sorted True for ONNX support)
        with torch.no_grad():
            path_lengths = (path_types != self.type_pad_id).sum(0)
        packed_path_types = nn.utils.rnn.pack_padded_sequence(path_types_embed, path_lengths, enforce_sorted=False)

        # [num layers * num directions; total paths; rnn size]
        _, (h_t, _) = self.path_lstm(packed_path_types)
        # [total_paths; rnn size * num directions]
        encoded_paths = h_t[-self.num_directions :].transpose(0, 1).reshape(h_t.shape[1], -1)
        encoded_paths = self.dropout_rnn(encoded_paths)

        # [total_paths; 2 * embedding size + rnn size (*2)]
        concat = torch.cat([encoded_from_tokens, encoded_paths, encoded_to_tokens], dim=-1)

        # [total_paths; output size]
        concat = self.embedding_dropout(concat)
        output = self.linear(concat)
        output = self.norm(output)
        output = torch.tanh(output)

        return output