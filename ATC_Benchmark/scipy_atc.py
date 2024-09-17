import torch
import torch.nn as nn
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage as linkage_scipy

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

        upper_traingle_indexes = np.triu_indices(T, k=1)
        scores = (1 - scores).cpu().numpy()
        cluster_labels = np.zeros((B,T),dtype=np.int64)
        label_dummy = np.zeros(T, dtype=np.int64)
        for b_idx in range(B):
            Z = linkage_scipy(scores[b_idx][upper_traingle_indexes], metric="cosine", method = linkage)
            scipy_labels = fcluster(Z, t=Z[T-num_clusters-1, 2], criterion="maxclust")
            cluster_labels[b_idx] = label_dummy
            
        cluster_labels = torch.from_numpy(cluster_labels).to(device = metric.device)