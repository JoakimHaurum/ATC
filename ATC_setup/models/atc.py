import numpy as np

import logging
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import PatchEmbed, Mlp, DropPath

from sklearn.cluster import AgglomerativeClustering

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }




class Attention_ToMe(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size=None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        metric = k.mean(1) 

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        
        x = self.proj(x)
        x = self.proj_drop(x)
         
        return x, metric


class Block_ATC(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_clusters = 1, linkage = "average", cls_token = True, dist_token = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_ToMe(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.num_clusters = num_clusters
        self.linkage = linkage
        self.cls_token = cls_token
        self.dist_token = dist_token

    def forward(self, x, attn_size = None):
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path(x_attn)
        
        cluster_assignment = None
        if self.num_clusters > 0:
            # Apply ToMe here
            merge, _, cluster_assignment = agglomerative_clustering(
                metric,
                self.num_clusters,
                self.linkage,
                self.cls_token,
                self.dist_token
            )
            x, attn_size = merge_wavg(merge, x, attn_size)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_size, cluster_assignment


class AgglomerativeClusteringVisionTransformer(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,  representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, 
                 act_layer=None, weight_init='', args=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__(img_size = img_size, 
                         patch_size = patch_size,
                         in_chans = in_chans,
                         num_classes = num_classes, 
                         embed_dim = embed_dim, 
                         depth = depth,
                         num_heads = num_heads,
                         mlp_ratio = mlp_ratio, 
                         qkv_bias = qkv_bias, 
                         drop_rate = drop_rate, 
                         attn_drop_rate = attn_drop_rate, 
                         drop_path_rate = drop_path_rate, 
                         embed_layer = embed_layer, 
                         norm_layer = norm_layer,
                         act_layer = act_layer, 
                         weight_init = weight_init)


        token_ratio = args.reduction_ratio
        reduction_loc = args.reduction_loc
        linkage = args.linkage
        
        if len(token_ratio) == 1:
            token_ratio = [int(self.patch_embed.num_patches * token_ratio[0] ** (idx+1)) for idx in range(len(reduction_loc))]
        
        assert len(token_ratio) == len(reduction_loc), f"Mismatch between the reduction location ({reduction_loc}) and token ratios ({token_ratio})"
        print(token_ratio, reduction_loc)


        token_ratio_full = [0 for _ in range(depth)]
        for idx, loc in enumerate(reduction_loc):
            token_ratio_full[loc] = token_ratio[idx]

        del(self.blocks)
        self.num_patches = self.patch_embed.num_patches
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_ATC(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_clusters=token_ratio_full[i], linkage=linkage)
            for i in range(depth)])

        self.deit_distillation = distilled

        self.reduction_loc = reduction_loc
        self.token_ratio = token_ratio
        self.prop_attn = args.proportional_attn
        
        self.viz_mode = getattr(args, 'viz_mode', False)

        self.apply(self._init_weights)

    def get_new_module_names(self):
        return []

    def get_reduction_count(self):
        return self.reduction_loc

    def forward(self, x):
        
        attn_size = None

        B = x.shape[0]
        x = self.patch_embed(x)
        
        if self.viz_mode:
            assignments = {}
        
        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks        
        x = torch.cat((cls_token, x), dim=1)
        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)

        for i, blk in enumerate(self.blocks):
            x, attn_size, cluster_assign = blk(x, attn_size)
            
            if self.viz_mode and i in self.reduction_loc:
                assignments[i] = cluster_assign.clone().detach().cpu().numpy()

            if not self.prop_attn:
                attn_size = None

        x = self.norm(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head(x)

        
        if self.viz_mode:
            viz_data = {"Assignment_Maps": assignments}
            return x, viz_data
        else:
            return x
            

def agglomerative_clustering(
    metric: torch.Tensor,
    num_clusters: int,
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

    B, T, _ = metric.shape

    num_clusters = min(num_clusters, T-protected)

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
    

    return merge, None, cluster_labels
    
    
def merge_wavg(
    merge, x: torch.Tensor, size: torch.Tensor = None
):
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


if __name__ == "__main__":

    batch_size = 1024
    num_tokens = 10#14*14
    feature_size = 8#512

    n_clusters = 3
    distance_threshold = None

    features = torch.randint(-10, 10, (batch_size, num_tokens, feature_size)).float()

    att = Attention_ToMe(feature_size, 2)

    x_attn, metric = att(features)

    attn_size = None

    import time


    merge, _, _ = agglomerative_clustering(metric, n_clusters, class_token=True)

    print(x_attn)
    x, attn_size = merge_wavg(merge, x_attn, attn_size)

    print(x)
    print(attn_size)

    start = time.time()
    blck1 = Block_ATC(feature_size, 2, num_clusters=4, cls_token=False)
    print(time.time()-start)
    blck2 = Block_ATC(feature_size, 2, num_clusters=2, cls_token=False)

    print(features.shape)
    x, attn_size = blck1(features, None)
    print(x.shape, attn_size)
    x, attn_size = blck2(x, None)
    print(x.shape, attn_size)
