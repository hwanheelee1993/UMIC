"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for ITM model
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
#from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from torch.nn import LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel
import numpy as np


class UniterForCaptioningMetric(UniterPreTrainedModel):
    """ Finetune UNITER for Caption Evaluation
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.margin = margin
        self.apply(self.init_weights)

    def init_output(self):
        """ need to be called after from pretrained only for the training step"""
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def forward(self, batch, compute_loss=True, compute_step_loss=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)

        if compute_loss:
            # triplet loss
            rank_scores_sigmoid = torch.sigmoid(rank_scores)
            sample_size = batch['sample_size']
            scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)

            pos = scores[:, :1]
            neg = scores[:, 1:]
          
            rank_loss = torch.clamp(self.margin + neg - pos, 0)
            
            #print("## Rank Score Sigmoid Size: ", rank_scores_sigmoid.size())
            #print("## Scores size: ", scores.size())  

            return rank_loss, rank_scores
        else:
            return rank_scores



class UniterForCaptionEvaluationLinearBCE(UniterPreTrainedModel):
    """ Finetune UNITER for Caption Evaluation
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        ce_scores = self.itm_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            ce_loss = F.binary_cross_entropy_with_logits(
                ce_scores, targets, reduction='none')
            return ce_loss
        else:
            return ce_scores

class UniterForCaptionEvaluationLinearRank(UniterPreTrainedModel):
    """ Finetune UNITER for Caption Evaluation
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.margin = margin
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True, is_val=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)

        if compute_loss:
            if(is_val):
                rank_scores_sigmoid = torch.sigmoid(rank_scores)
            else:
                # triplet loss
                rank_scores_sigmoid = torch.sigmoid(rank_scores)
                sample_size = batch['sample_size']
                scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)

                pos = scores[:, :1]
                neg = scores[:, 1:]
                
                rank_loss = torch.clamp(self.margin + neg - pos, 0)
                
                return rank_loss, rank_scores
        else:
            return rank_scores
