
import argparse
import os
import pandas as pd
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.transforms.functional import InterpolationMode

import timm
from timm.utils import accuracy, AverageMeter, setup_default_logging

import SSLBackbone_setup.token_reduc as token_reduc
import SSLBackbone_setup.models_vit_mae as mae_models


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--data', type=str, help='dataset path')
parser.add_argument('--split', metavar='NAME', default='val',
                    help='dataset split (default: val)')
parser.add_argument('--model-type', help='Model stype [AugReg, SWAG, MAE]', default="augreg", choices=["augreg", "swag", "mae"])
parser.add_argument('--model-name', '-m', metavar='NAME', help='backbone model', default="vit_tiny_patch16_224")
parser.add_argument('--input-size', type=int, default=224)
parser.add_argument('--r', type=int, nargs='+', default=[0]) 
parser.add_argument('--r-sched', type=str, default="constant", choices=["constant", "decrease"]) 
parser.add_argument('--merge_fn', type=str, default="tome", choices=["tome", "atc"]) 
parser.add_argument('--linkage', type=str, default="average", choices=["average","single","complete"]) 
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use_amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-dir', default='./tome_notrain_results', type=str, metavar='DIRBANE',
                    help='Output directory for validation results (summary)')

# AugReg models:
## vit_tiny_patch16_224 -- r [0, 16] -- im 224
## vit_small_patch16_224 -- r [0, 16] -- im 224
## vit_base_patch16_224 -- r[0, 16] -- im 224
## vit_large_patch16_224 -- r[0, 8] -- im 224
## vit_large_patch16_384 -- r [0, 5 ,10, 15, 20, 23] -- im 384
def timm_prep(model_name = "vit_base_patch16_224", input_size = 224):
    # Load a pretrained model
    model = timm.create_model(model_name, pretrained=True)

    input_size = model.default_cfg["input_size"][1]

    transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
    ])

    return model, transform, token_reduc.patch.timm, (model.default_cfg["mean"], model.default_cfg["std"])

# SWAG models:
## vit_b16_in1k -- r [0, 5, 10, 15, 20, 25, 30, 35, 40, 45] -- im 384
## vit_l16_in1k -- r [0, 5, 10, 15, 20, 25, 30, 35, 40] -- im 512
## vit_h14_in1k -- r [0, 5, 10, 15, 20, 25, 30, 35, 40] -- im 518
def swag_prep(model_name = "vit_b16_in1k", input_size = 384):
    # Load a pretrained model
    model = torch.hub.load("facebookresearch/swag", model=model_name)

    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, token_reduc.patch.swag, ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# MAE models:
## vit_base_patch16 -- r [0, 16] -- im 224
## vit_large_patch16 -- r [0, 8] -- im 224
## vit_huge_patch14 -- r [0, 7] -- im 224
def mae_prep(model_name = "vit_base_patch16", input_size = 224):
    model = mae_models.__dict__[model_name](
            num_classes=1000,
            drop_path_rate=0.0,
            global_pool=True,
        )
    
    model = mae_models.prepare_model(model, model_name, "./pretrained_models")
    
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    
    size = int(input_size / crop_pct)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    return model, transform, token_reduc.patch.mae, ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



def main():
    setup_default_logging()
    args = parser.parse_args()
    amp_autocast = suppress  # do nothing
    if args.use_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')

    device = torch.device(args.device)

    if args.model_type == "augreg":
        model, transform, patch_fn, norm = timm_prep(args.model_name, args.input_size)
    elif args.model_type == "swag":
        model, transform, patch_fn, norm = swag_prep(args.model_name, args.input_size)
    elif args.model_type == "mae":
        model, transform, patch_fn, norm = mae_prep(args.model_name, args.input_size)

    patch_fn(model, args.merge_fn, args.linkage)
    model.r = 0

    dataset_val = ImageNet(root=args.data, split=args.split, transform=transform)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
       
    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info(f'Model {args.model_type} {args.model_name} created, param count: {param_count}')

    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss().to(device)

    args.results_dir = os.path.join(args.results_dir, args.r_sched)

    os.makedirs(args.results_dir, exist_ok=True)

    if args.results_file == "":
        args.results_file = f"{args.model_type}_{args.model_name}_{args.input_size}_{args.merge_fn}"
        if args.merge_fn != "tome":
            args.results_file += f"_{args.linkage}"
        args.results_file += ".csv"

    results_dict = {"Model":[],
                    "Img_Size":[],
                    "r":[],
                    "Top-1":[],
                    "Top-5":[],
                    "Param":[]}
    for r in args.r:
        if r != 0:
            if args.r_sched == "constant":
                model.r = r
            elif args.r_sched == "decrease":
                model.r = (r, -1)

        _logger.info(f"{model.r}, {r}")
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        model.eval()

        with torch.no_grad():
            end = time.time()
            for batch_idx, (input, target) in enumerate(data_loader_val):
                target = target.to(device, non_blocking=True)
                input = input.to(device, non_blocking=True)

                # compute output
                with amp_autocast():
                    output = model(input)
                    loss = criterion(output, target)

                losses.update(loss.item(), input.size(0))

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    _logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                        'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                            batch_idx, len(data_loader_val), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses, top1=top1, top5=top5))

        top1a, top5a = top1.avg, top5.avg
        results = OrderedDict(
            model = args.model_name, img_size = args.input_size,
            top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
            top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
            param_count=round(param_count / 1e6, 2))

        _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
        results['top1'], results['top1_err'], results['top5'], results['top5_err']))

        results_dict["Model"].append(args.model_name)
        results_dict["Img_Size"].append(args.input_size)
        results_dict["r"].append(r)
        results_dict["Top-1"].append(top1a)
        results_dict["Top-5"].append(top5a)
        results_dict["Param"].append(round(param_count / 1e6, 2))
    
        df_results = pd.DataFrame(results_dict)
        df_results.to_csv(os.path.join(args.results_dir, args.results_file), sep=";", index=False)

        

if __name__ == '__main__':
    main()
