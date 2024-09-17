# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import numpy as np
from typing import Callable, Tuple
from sklearn.cluster import AgglomerativeClustering

import torch


def do_nothing(x, mode=None):
    return x


def agglomerative_clustering(
    metric: torch.Tensor,
    r: int,
    linkage: str = "average",
    class_token: bool = False,
    distill_token: bool = False,
):
    """
    Input size is [batch, tokens, channels].
    num_clusters indicates the number of clusters to construct 
    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    B, T, C = metric.shape
    
    r = min(r, (T - protected))
    num_clusters = T-r

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        scores = metric @ metric.transpose(-1, -2)

        if class_token:
            scores = scores[:, 1:, 1:]
            T -= 1

        #upper_traingle_indexes = np.triu_indices(T, k=1)
        scores = (1 - scores).cpu().numpy()
        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", linkage=linkage, distance_threshold=None)

        cluster_labels = np.zeros((B,T),dtype=np.int64)
        for b_idx in range(B):
            labels = clustering.fit(scores[b_idx]).labels_
            #Z = linkage_fns(scores[b_idx][upper_traingle_indexes], method = linkage)
            #labels = fcluster(Z, t=Z[T-num_clusters-1, 2], criterion="distance") - 1
            cluster_labels[b_idx] = labels
            
        cluster_labels = torch.from_numpy(cluster_labels).to(device = metric.device)

        if class_token:
            # Sort to ensure the class token is at the start
            cluster_labels = cluster_labels + protected
            cluster_labels = torch.cat([torch.zeros(B, 1, device = metric.device).long(), cluster_labels], dim=-1)

   
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        C = x.shape[-1]
        dst  = torch.zeros(B, num_clusters+protected, C, device=x.device)
        dst = dst.scatter_reduce(-2, cluster_labels.unsqueeze(-1).repeat(1,1,C), x, reduce=mode)
        return dst
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        first_index = cluster_labels[:, :num_clusters].unsqueeze(-1).expand(B, num_clusters, C)
        second_index = cluster_labels[:, num_clusters:].unsqueeze(-1).expand(B, r, C)

        first_half_out = torch.gather(x, dim=-2, index=first_index)
        second_half_out = torch.gather(x, dim=-2, index=second_index)
        out = torch.concat((first_half_out, second_half_out), dim=1)

        return out

    return merge, unmerge
    

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    linkage: str = "average",
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
