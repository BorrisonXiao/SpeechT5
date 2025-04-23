# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import torch.nn as nn
import torch


logger = logging.getLogger(__name__)

class SpeechEncoderPostnet(nn.Module):
    """

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(self, dictionaries, args):
        super(SpeechEncoderPostnet, self).__init__()
        # modules below are not needed during fine-tuning
        self.target_glu = args.target_glu
        self.skip_masked = args.skip_masked
        self.skip_nomask = args.skip_nomask
        self.logit_temp = args.logit_temp

        final_dim = (
            args.final_dim if args.final_dim > 0 else args.encoder_embed_dim
        )
        if any([d is None for d in dictionaries]):
            logger.info(
                "cannot find dictionary. assume will be used for fine-tuning"
            )
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)
        self.untie_final_proj = args.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                args.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(args.encoder_embed_dim, final_dim)

    # def compute_nce(self, x, pos, negs):
    #     neg_is_pos = (pos == negs).all(-1)
    #     pos = pos.unsqueeze(0)
    #     targets = torch.cat([pos, negs], dim=0)
    #     logits = torch.cosine_similarity(
    #         x, targets.type_as(x), dim=-1
    #     )
    #     logits /= self.logit_temp
    #     if neg_is_pos.any():
    #         logits[1:][neg_is_pos] = float("-inf")
    #     logits = logits.transpose(0, 1)  # (num_x, num_cls+1)

    #     return logits
    
    def compute_nce(self, x, pos, negs, chunk_size=512):
        """
        Process instances in chunks to reduce memory fragmentation.
        Args:
            x:      (num_x, embed_dim)         # Input embeddings
            pos:    (num_x, embed_dim)      # Positive targets
            negs:   (num_cls, num_x, embed_dim) # Negative targets
            chunk_size: Fixed chunk size for `num_x` dimension.
        Returns:
            logits: (num_x, num_cls + 1)       # NCE logits
        """
        num_x, embed_dim = x.shape
        num_cls = negs.shape[0]
        logits = []

        pos = pos.unsqueeze(0)  # (1, num_x, embed_dim)

        # Process chunks of instances to limit temporary memory
        for i in range(0, num_x, chunk_size):
            # Slice current chunk of instances
            x_chunk = x[i:i+chunk_size]  # (chunk_size, embed_dim)
            pos_chunk = pos[:, i:i+chunk_size]  # (1, chunk_size, embed_dim)
            negs_chunk = negs[:, i:i+chunk_size]  # (num_cls, chunk_size, embed_dim)

            # Concatenate pos + negs for the chunk
            targets_chunk = torch.cat([pos_chunk, negs_chunk], dim=0)  # (num_cls+1, chunk_size, embed_dim)

            # Compute cosine similarity between x_chunk and targets_chunk
            logits_chunk = torch.cosine_similarity(
                x_chunk.unsqueeze(0),  # (1, chunk_size, embed_dim)
                targets_chunk.type_as(x_chunk),  # (num_cls+1, chunk_size, embed_dim)
                dim=-1
            )  # (num_cls+1, chunk_size)

            logits_chunk /= self.logit_temp

            # Mask where negatives == positive for this chunk
            neg_is_pos_chunk = (pos_chunk == negs_chunk).all(-1)  # (num_cls, chunk_size)
            if neg_is_pos_chunk.any():
                logits_chunk[1:][neg_is_pos_chunk] = float("-inf")

            # Transpose to (chunk_size, num_cls + 1)
            logits.append(logits_chunk.transpose(0, 1))
            maybe_empty_cache()

        # Combine all chunks (final shape: num_x, num_cls + 1)
        return torch.cat(logits, dim=0)

    def forward(self, x, padding_mask, mask_indices, target_list, pad: int = -100):
        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            # # Cihan: For padded tokens, the target is -100, so we need to filter them out
            # _target = target[target != -100]
            assert (target >= 0).all(), f"target has negative values: {target}"
            y = torch.index_select(label_embs, 0, target.long())
            # Pad y with zeros to match the shape of proj_x
            y = torch.cat([y, torch.zeros(proj_x.size(0) - y.size(0), y.size(1)).to(y.device)], dim=0)
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(
                    zip(proj_x_m_list, target_list)
                )
            ]
            # compute_pred(proj_x_m_list[0], target_list[0][masked_indices], label_embs_list[0])
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(
                    zip(proj_x_u_list, target_list)
                )
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
        }

        return result

def maybe_empty_cache(limit_mib=36000, verbose=False):
    reserved_bytes = torch.cuda.memory_reserved()
    reserved_mib = reserved_bytes / (1024 ** 2)
    
    if reserved_mib > limit_mib:
        if verbose:
            print(f"[Reserved] {reserved_mib} MiB exceeds limit ({limit_mib} MiB). Calling torch.cuda.empty_cache()...")
            # print(f"[Action] Exceeds limit ({limit_mib} MiB). Calling torch.cuda.empty_cache()...")
        torch.cuda.empty_cache()