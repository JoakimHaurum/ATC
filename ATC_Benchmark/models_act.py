# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial

from timm.models.deit import _cfg, default_cfgs
from timm.models.registry import register_model


__all__ = [
    'deit_tiny_patch16_224_local', \
    'deit_small_patch16_224_local', \
    'deit_base_patch16_224_local', \
    "tome_tiny_patch16_224", \
    "tome_small_patch16_224", \
    "tome_base_patch16_224", \
    "atc_tiny_patch16_224", \
    "atc_small_patch16_224", \
    "atc_base_patch16_224", \
    "kmedoids_tiny_patch16_224", \
    "kmedoids_small_patch16_224", \
    "kmedoids_base_patch16_224", \
    "dpcknn_tiny_patch16_224", \
    "dpcknn_small_patch16_224", \
    "dpcknn_base_patch16_224", \
]


deit_url_paths = {"deit_tiny_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                  "deit_tiny_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
                  "deit_small_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                  "deit_small_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
                  "deit_base_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                  "deit_base_distilled_patch16_224": "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
                  }


@register_model
def deit_tiny_patch16_224_local(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from timm.models.vision_transformer import VisionTransformer

    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]
    if pretrained:

        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )

        print(checkpoint["model"].keys())
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def deit_small_patch16_224_local(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from timm.models.vision_transformer import VisionTransformer

    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not A-ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_base_patch16_224_local(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from timm.models.vision_transformer import VisionTransformer
    
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    kwargs.pop("args", None)

    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def tome_tiny_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.tome import ToMeVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ToMeVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def tome_small_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.tome import ToMeVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ToMeVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def tome_base_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.tome import ToMeVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = ToMeVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    





@register_model
def atc_tiny_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.static_agg import AgglomerativeClusteringVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = AgglomerativeClusteringVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def atc_small_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.static_agg import AgglomerativeClusteringVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = AgglomerativeClusteringVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def atc_base_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.static_agg import AgglomerativeClusteringVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = AgglomerativeClusteringVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
    

@register_model
def kmedoids_tiny_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.kmedoids import KMedoidsVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = KMedoidsVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def kmedoids_small_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.kmedoids import KMedoidsVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = KMedoidsVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def kmedoids_base_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.kmedoids import KMedoidsVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = KMedoidsVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model

    


@register_model
def dpcknn_tiny_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.dpcknn import DPCKNNVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = DPCKNNVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if deit_distillation:
        key = "deit_tiny_distilled_patch16_224"
    else:
        key = "deit_tiny_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def dpcknn_small_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.dpcknn import DPCKNNVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = DPCKNNVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_small_distilled_patch16_224"
    else:
        key = "deit_small_patch16_224"

    model.default_cfg = default_cfgs[key]
    
    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


@register_model
def dpcknn_base_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):

    from ..ICCV_Benchmark.models.dpcknn import DPCKNNVisionTransformer
    
    if hasattr(kwargs["args"], "distillation_type"):
        deit_distillation = kwargs["args"].distillation_type != 'none'
    else: 
        deit_distillation = False

    model = DPCKNNVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if deit_distillation:
        key = "deit_base_distilled_patch16_224"
    else:
        key = "deit_base_patch16_224"

    model.default_cfg = default_cfgs[key]

    if pretrained:
        # note that this part loads DEIT weights, not Dynamic ViTs
        checkpoint = torch.hub.load_state_dict_from_url(
            url=deit_url_paths[key],
            model_dir = "./deit_weights",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
