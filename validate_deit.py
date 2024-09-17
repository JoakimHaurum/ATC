
import os
import csv
import time
import argparse

import numpy as np

from json import JSONEncoder
from contextlib import suppress
from collections import OrderedDict
import ATC_setup.utils as utils

import torch

from timm.models import create_model
from timm.utils import accuracy, AverageMeter

from ATC_setup.datasets import build_dataset
import ATC_setup.models_act as models_act

try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Validating trained methods')
parser.add_argument('--data', metavar="DIR", type=str, help='dataset path')
parser.add_argument('--dataset', '-d', metavar='NAME', default='imagenet', choices=['imagenet', 'nabirds', "coco", "nuswide"], type=str, help='Image Net dataset path')
parser.add_argument('--split', metavar='NAME', default='validation', help='dataset split (default: validation)')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
parser.add_argument('--val_output_name', default='', help='file neame for resulting validation data')
parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use_amp', action='store_true', help="")
parser.add_argument('--device', default='cuda', help='device to use for training / testing')

parser.add_argument('--ots', action='store_true', help="")
temp_args, _ = parser.parse_known_args()

if temp_args.ots:
    parser.add_argument('--model', default='atc_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--reduction_loc', type=int, nargs='+', default=[]) 
    parser.add_argument('--reduction_ratio', type=float, nargs='+', default=[])
        
    if "atc" in temp_args.model.lower():
        parser.add_argument('--linkage', default="average", type=str)
        parser.add_argument('--proportional_attn', action="store_true")


def validate(args, _logger):
    amp_autocast = suppress  # do nothing
    if args.use_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')

    assert args.checkpoint != "", "Empty checkpoint path, not usable"
    assert os.path.isfile(args.checkpoint), "Checkpoint path is not file, not usable: {}".format(args.checkpoint)
           

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Setting for posterity
    args.color_jitter = 0
    args.aa = ""
    args.train_interpolation = "bicubic"
    args.reprob = 0
    args.remode = ""
    args.recount = 0

    dataset_val, args.num_classes = build_dataset(args.data, args.dataset, "val", args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_args = checkpoint["args"]
    
    if args.ots:
        _logger.info(f"Creating model: {args.model}")
        model_args = args
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size,
            args = model_args
        )
    else:
        _logger.info(f"Creating model: {model_args.model}")
        model = create_model(
            model_args.model,
            pretrained=False,
            num_classes=args.num_classes,
            img_size=model_args.input_size,
            args = model_args
        )

    if checkpoint["ema_best"]:
        model.load_state_dict(checkpoint['model_ema'])
    else:
        model.load_state_dict(checkpoint['model'])

    
    _logger.info("counting parameters")

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("logging")
    _logger.info('Model %s created, param count: %d' % (model_args.model, param_count))

    _logger.info("moving to device")
    model.to(device)
    model.eval()

    

    _logger.info("Setting up Loss")

    if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    metric_logger = utils.MetricLogger(delimiter="  ")

    
    model.eval()

    model_name = model_args.model

    model_args_dict = vars(model_args)
    linkage = model_args_dict.get("linkage", ["UNK"])

    model_data_dict = {"Model": model_name,
                       "Ratio": model_args.reduction_ratio,
                       "Linkage": linkage,
                       "Location": model_args.reduction_loc}

    if args.dataset.lower() == "imagenet":
        image_names = [os.path.basename(s[0]) for s in dataset_val.samples]
    elif args.dataset.lower() == "nabirds":
        image_names = [dataset_val.data.iloc[idx].img_id for idx in range(len(dataset_val))]
    elif args.dataset.lower() == "coco":
        image_names = [dataset_val.ids[idx] for idx in range(len(dataset_val))]
    elif args.dataset.lower() == "nuswide":
        image_names = [os.path.splitext(os.path.basename(x[0]))[0] for x in dataset_val.itemlist]

    _logger.info("Ready for Inference")

    if args.dataset.lower() == "coco" or args.dataset.lower() == "nuswide":            
        Sig = torch.nn.Sigmoid()
        preds_regular = []
        targets = []


    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        #input = torch.randn((args.batch_size,) + tuple(3, args.input_size, args.input_size)).to(device)
        #model(input)
        end = time.time()

        img_count = 0
        for batch_idx, (input, target) in enumerate(data_loader_val):
            target = target.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)

            # compute output
            with amp_autocast():
                output = model(input)

            if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
                loss = criterion(output, target)
            elif args.dataset.lower() == "coco":
                target = target.max(dim=1)[0].float()
                output = output.float()
                loss = criterion(output, target)
            elif args.dataset.lower() == "nuswide":
                loss = criterion(output.float(), target.float())
            
            batch_size = input.shape[0]
            metric_logger.update(loss=loss.item())
            losses.update(loss.item(), input.size(0))
            
            if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                _, pred = output.topk(5, 1, True, True)
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))
                
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            else:
                # Measure mAP
                pred = Sig(output)
                preds_regular.append(pred.cpu().detach())
                targets.append(target.cpu().detach())

            for i in range(target.shape[0]):
                image_name = image_names[img_count + i]

                data_dict = {"Predictions": pred[i].cpu().numpy(),
                             "Target": target[i].cpu().numpy(),
                             "Loss": loss.item()}
                model_data_dict[image_name] = data_dict
            img_count += target.shape[0]


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 20 == 0:
                if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
                    _logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                        'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'
                        'ML_Loss: {MLloss.value:>7.4f} ({MLloss.global_avg:>6.4f})  '
                        'ML_Acc@1: {MLtop1.value:>7.3f} ({MLtop1.global_avg:>7.3f})  '
                        'ML_Acc@5: {MLtop5.value:>7.3f} ({MLtop5.global_avg:>7.3f})'.format(
                            batch_idx, len(data_loader_val), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses, top1=top1, top5=top5,
                            MLloss=metric_logger.loss, MLtop1=metric_logger.acc1, MLtop5=metric_logger.acc5))
                else:
                        _logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'ML_Loss: {MLloss.value:>7.4f} ({MLloss.global_avg:>6.4f})  '.format(
                            batch_idx, len(data_loader_val), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses, MLloss=metric_logger.loss))
    
        
    if args.dataset.lower() == "coco" or args.dataset.lower() == "nuswide":
        mAP_score = utils.mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
        metric_logger.meters['acc1'].update(mAP_score, n=1)
        metric_logger.meters['acc5'].update(mAP_score, n=1)
        top1.update(mAP_score, 1)
        top5.update(mAP_score, 1)


    top1a, top5a = top1.avg, top5.avg
    MLtop1a, MLtop5a = metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        MLtop1=round(MLtop1a, 4), MLtop1_err=round(100 - MLtop1a, 4),
        MLtop5=round(MLtop5a, 4), MLtop5_err=round(100 - MLtop5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=args.input_size)

    model_data_dict["Top1-Acc"] = round(top1a, 4)
    model_data_dict["Top5-Acc"] = round(top5a, 4)
    model_data_dict["MLTop1-Acc"] = round(MLtop1a, 4)
    model_data_dict["MLTop5-Acc"] = round(MLtop5a, 4)
    model_data_dict["Params"] = round(param_count / 1e6, 2)

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})\n * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err'], results['MLtop1'], results['MLtop1_err'], results['MLtop5'], results['MLtop5_err']))

    return model_data_dict


def main(args, _logger):  
    
    val_data = validate(args, _logger)
    val_data_file = os.path.join(args.output_dir, args.val_output_name)
    write_val(val_data_file, val_data)

def write_val(val_file, val_data):
    with open(val_file, "w") as write_file:
        json.dump(val_data, write_file, cls=NumpyArrayEncoder, indent=4)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()

if __name__ == '__main__':

    from timm.utils import setup_default_logging
    import logging

    _logger = logging.getLogger('validate')
    setup_default_logging()
    args = parser.parse_args()
    main(args, _logger)
