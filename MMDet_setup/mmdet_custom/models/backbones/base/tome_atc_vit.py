import numpy as np
import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from .vit import PatchEmbed, Mlp, DropPath

from sklearn.cluster import AgglomerativeClustering
#from scipy.cluster.hierarchy import fcluster
#from scipy.cluster.hierarchy import linkage as linkage_fns

from functools import partial


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

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, k.mean(1)


class Block_ToMeATC(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_ratio = 1, 
                 linkage = "average", cls_token = False, dist_token = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_ToMe(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.linkage = linkage
        self.cls_token = cls_token
        self.dist_token = dist_token

        print(f"Block CLS Token {self.cls_token}")

        if self.linkage == "tome":
            self.merge_fn = self.tome_forward
        else:
            self.merge_fn = self.atc_forward
        self.token_ratio = token_ratio

    def tome_forward(self, metric):
        T = metric.shape[1]
        num_clusters = int(self.token_ratio*T)
        r = T-num_clusters
        merge, unmerge = bipartite_soft_matching(
                metric,
                r,
                self.cls_token,
                self.dist_token
            )
        
        return merge, unmerge
    
    def atc_forward(self, metric):
        T = metric.shape[1]
        num_clusters = int(self.token_ratio*T)
        merge, unmerge = agglomerative_clustering(
                metric,
                num_clusters,
                self.linkage,
                self.cls_token,
                self.dist_token
            )
        
        return merge, unmerge

    def forward(self, x, attn_size = None):
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path(x_attn)
        
        merge = None
        unmerge = None
        if self.token_ratio > 0:
            # Apply ToMe here
            merge, unmerge = self.merge_fn(metric)
            x, attn_size = merge_wavg(merge, x, attn_size)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_size, merge, unmerge


class ToMeATCVisionTransformer(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,  representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, 
                 act_layer=None, weight_init='', reduction_ratio=0.9, reduction_loc=[3,6,9], linkage="tome", proportional_attn=True,
                 layer_scale=False, window_attn=False, window_size=None, pretrained=False):
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

        token_ratio = reduction_ratio
        pruning_loc = reduction_loc
        linkage = linkage
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.pretrain_size = img_size

        if not isinstance(token_ratio, (list, tuple)):
            token_ratio = [token_ratio]
        
        if len(token_ratio) == 1:
            token_ratio = [token_ratio[0] for _ in range(len(pruning_loc))]
        
        assert len(token_ratio) == len(pruning_loc), f"Mismatch between the pruning location ({pruning_loc}) and token ratios ({token_ratio})"
        print(token_ratio, pruning_loc)

        token_ratio_full = [0 for _ in range(depth)]
        for idx, loc in enumerate(pruning_loc):
            token_ratio_full[loc] = token_ratio[idx]
            

        del(self.blocks)
        self.num_patches = self.patch_embed.num_patches
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm_layer = norm_layer
        self.act_layer = act_layer        


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_ToMeATC(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, token_ratio=token_ratio_full[i], linkage=linkage)
            for i in range(depth)])

        self.deit_distillation = distilled

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        self.prop_attn = proportional_attn

        self.apply(self._init_weights)

    def get_new_module_names(self):
        return []

    def get_reduction_count(self):
        return self.pruning_loc

    def forward(self, x):
        
        attn_size = None

        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks        
        x = torch.cat((cls_token, x), dim=1)
        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)

        for i, blk in enumerate(self.blocks):
            x, attn_size, unmerge = blk(x, attn_size)

            if not self.prop_attn:
                attn_size = None

        x = self.norm(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head(x)

        return x
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}


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
        C = x.shape[-1]
        r = T+protected-num_clusters
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
    class_token: bool = False,
    distill_token: bool = False,
):
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
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)

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
