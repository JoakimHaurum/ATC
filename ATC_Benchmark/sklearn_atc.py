import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering

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
    linkage: str = "average"
):
    """
    Input size is [batch, tokens, channels].
    num_clusters indicates the number of clusters to construct 
    """
    protected = 0

    B, T, _ = metric.shape

    num_clusters = min(num_clusters, T-protected)

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        scores = metric @ metric.transpose(-1, -2)

        scores = (1 - scores).cpu().numpy()
        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", linkage=linkage, distance_threshold=None)

        cluster_labels = np.zeros((B,T),dtype=np.int64)
        for b_idx in range(B):
            labels = clustering.fit(scores[b_idx]).labels_
            cluster_labels[b_idx] = labels
            
        cluster_labels = torch.from_numpy(cluster_labels).to(device = metric.device)