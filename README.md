# Agglomerative Token Clustering

![Agglomerative Token Clustering (ATC) method builds clusters locally, i.e. by iteratively combining the most similar tokens, until the desired amount of tokens remain. A step of this process is shown here, where a graph of nodes (in this case, tokens) are connected with edges based on their similarity. The most similar pair of nodes (highlighted in blue) are combined, and the edges are updated using linkage function D, in this case the "complete" linkage function.](./repo_images/ATC_steps.JPG)

This repository is the official implementation of [Agglomerative Token Clustering](). 

We present Agglomerative Token Clustering (ATC), a novel token merging method that consistently outperforms previous token merging and pruning methods across image classification, image synthesis, and object detection & segmentation tasks. ATC merges clusters through bottom-up hierarchical clustering, without the introduction of extra learnable parameters.

The project page can be found [here](http://vap.aau.dk/atc).

## Requirements

Requirements for each experiment/setup can be found in the `requirements` folder. 

## Datasets

We test on the ImageNet-1K, NABirds, COCO 2014, and NUS-WIDE datasets. They are available through the following links:

- ImageNet is available through Kaggle: [https://www.kaggle.com/c/imagenet-object-localization-challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge)
- NABirds is available through the official website: [https://dl.allaboutbirds.org/nabirds](https://dl.allaboutbirds.org/nabirds)
- COCO is available through the official website: [https://cocodataset.org/](https://cocodataset.org/)
- For NUS-WIDE, we use the variation made available by Alibaba-MIIL: [https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md](https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md)


## Fine-tuning DeiT model

These models are trained by finetuning a pretrained DeiT model. Reduction blocks are inserted using the `--reduction_loc` argument. In our paper we focus on reducing at the 4th, 7th, and 10th block (note that the argument is 0 indexed), as commonly done in the litterature. In order to work with a larger effective batch size than what can be fitted on the GPU VRAM, we use the `--grad_accum_steps` argument to aggregate over batches.

An example training command is shown below, training on ImageNet with the ATC method and a keep rate of 0.9.

```
train_deit.py --dataset imagenet --data <path_to_data> --batch-size 256 --lr 0.001 --epochs 30 --warmup-epochs 20 --lr_batch_normalizer 1024 --sched_in_steps --use_amp --grad_accum_steps 2 --wandb_project <wandb_project> --wandb_group <wandb_group> --output_dir <path_to_model_output> --model atc_small_patch16_224 --linkage complete --proportional_attn --reduction_loc 3 6 9 --keep_rate 0.9
```

## Fine-tuning MAE model

In order to fine-tune the MAE models we use the default settings from the original paper. An example training command is shown below, training with the ATC method and a keep rate of 0.9. It is assumed the pretrianed MAE checkpoints have been downloaded and placed in the `pretrained_models` folder.


```
train_mae.py --data <path_to_data> --batch_size 256 --accum_iter 2 --model vit_base_patch16 --finetune ./pretrained_models/mae_pretrain_vit_base.pth --epochs 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --wandb_project <wandb_project> --wandb_group <wandb_group> --output_dir <path_to_model_output> --merge_fn atc --r-count 16 --r-sched constant --linkage average
```

## Evaluating Off-the-Shelf SSL Backbone performance

Following the off-the-shelf analysis of the ToMe paper, we compare perforamnce when applying token reduction at different strengths to MAE, SWAG, and AugReg pretrained ViT models. In this experiment the token reduction method is applied at eeach stage, with the number of tokens removed determined by the `--r` and `--r-sched` arguments.

In order to evaluate without any prior training we use the following command:

```
validate_ots_sslbackbone.py --data <path_to_data> --batch_size 256 --model-type swag --model-name vit_base_patch16 --input-size 384 --r 0 5 10 15 20 25 30 35 40 45 --merge_fn atc --linkage single --r-sched decrease
```


## Fine-tuning ViT-Adapter

For hte object detection/segmenetation experiments we follow the setup from the SVIT paper. The ViT-Adapter model has been adapted to use token reduction, and the model has been defined in the `MMDet_setup/configs` folder. The specific linkage function and reduction rate are supplied via the `--cfg-options` command.


An example training command is shown below, training a ViT-Adapter-T with ATC using the single linkage function and a reduction rate of 25%.

```
train_mmdet.py MMDet_setup/configs/mask_rcnn/tomeatc-adapter-t-0.5x-ftune.py --cfg-options model.backbone.linkage=single model.backbone.reduction_ratio=0.25 cumulative_iters=2 data.samples_per_gpu=8 work_dir=<path_to_model_output> --launcher pytorch --wandb_project <wandb_project> --wandb_group <wandb_group>
```

## Generating examples with Stable-Diffusion
As proposed in the ToMe-SD paper, token reduction can be used to minimize the number of tokens used in diffusion models such as Stable-Diffusion, without any retraining.


An example command is shown below, applying ATC with the averag elinkage functio and keeping 90% of the tokens (NOTE: In this script `--reduction_ratio` indicate how large a ratio of tokens should be removed).

```
sd_generate.py --merge_fn_str atc --reduction_ratio 0.1 --merge_attn --linkage average
```

In order to evaluate the quality of the SD generations, we compute the FID score between the generated images and a set of reference images from the ImageNet Validation set (obtained with the `create_im_valset.py` script in the `SD_setup` folder):

```
python compute_fid_scores.py --mode clean --im_val_dir <path_to_im_val_ref> --sd_gen_dir <path_to_sd_generations>
```

## Benchmarking inference throughput

In order to benchmark the inference throughput of the ATC token reduction method, we use two scripts.

To compare the inference throughput of hte full model, we use the `bechnmark_merging_method.py` script:


An example is shown below measuring the inference throughput with reduction ratio of 50% and multiple batch sizes (defined as the power of base 2). 


```
benchmark_merging_method.py --output_dir <path_to_output_folder >--model atc_small_patch16_224 --input_size 224 --runs 2000 --throw_out 0.25 --reduction_ratio 0.5 --batch_sizes 0 1 2 3 4 5 6 7 8 9 10 --linkage complete
```


To compare the infernce speed of the different agglomerative clustering implementations in scikit-learn, scipy, and RAPIDS, we use the `benchmark_atc_framework.py` script.

An example is shown below measuring the inference throughput with the scikit-learn package, with reduction ratio of 25% and multiple batch sizes (defined as the power of base 2) and input_sizes. 

```
benchmark_atc_framework.py --output_dir <path_to_output_folder> --framework sklearn --input_sizes 224 256 --emb_size 64 --runs 1000 --throw_out 0.25 --reduction_ratio 0.25 --batch_sizes 0 1 2 3 4 5 6 7 8 9 10 --linkage complete
```

## Code Credits

The token reduction method code is based and inspired by:
- The ToMe method is based on the official code repository: [https://github.com/facebookresearch/ToMe](https://github.com/facebookresearch/ToMe)
- The ToMe-SD method and SD setup is based on the official code repository: [https://github.com/facebookresearch/ToMe](https://github.com/facebookresearch/ToMe)
- The SViT method and ViT-Adapter object detection/segmentation setup is based on the official code repository: [https://github.com/uzh-rpg/svit/](https://github.com/uzh-rpg/svit/)

Parts of the training code and large part of the ViT implementation is based and inspired by:
- The timm framework by Ross Wrightman / HuggingFace: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- The DeiT training and evaluation code: [https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)
- The multi-label training and evaluation code from the Assymetric Loss paper: [https://github.com/Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL)


## License

The Code is licensed under an MIT License, with exceptions of the afforementioned code credits which follows the license of the original authors.

## Bibtex
```bibtex
@InProceedings{Haurum_2024_ECCV,
author = {Joakim Bruslund Haurum and Sergio Escalera and Graham W. Taylor and Thomas B. Moeslund},
title = {Agglomerative Token Clustering},
booktitle = {Computer Vision — ECCV 2024},
month = {October},
year = {2024}, }
```
