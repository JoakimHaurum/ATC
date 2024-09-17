import torch
import torch.nn as nn
import numpy as np
from cuml import AgglomerativeClustering


class Block_ATC(nn.Module):

    def __init__(self, num_clusters = 1, linkage = "average"):
        super().__init__()
        self.num_clusters = num_clusters
        self.linkage = linkage

    def forward(self, x):
        # Apply ToMe here
        agglomerative_clustering(
            x,
            self.num_clusters,
            self.linkage
        )


def agglomerative_clustering(
    metric: torch.Tensor,
    num_clusters: int,
    linkage: str = "single"
):
    """
    Input size is [batch, tokens, channels].
    num_clusters indicates the number of clusters to construct 
    """
    protected = 0

    B, T, _ = metric.shape

    num_clusters = min(num_clusters, T-protected)

    with torch.no_grad():
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity="cosine", connectivity="pairwise", output_type="numpy")

        cluster_labels = np.zeros((B,T),dtype=np.int64)
        for b_idx in range(B):
            labels = clustering.fit(metric[b_idx]).labels_
            cluster_labels[b_idx] = labels
            
        cluster_labels = torch.from_numpy(cluster_labels).to(device = metric.device)