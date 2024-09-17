#///////////// Tesing Images ////////////
test_imgs = ['./COCO/val2017/000000022935.jpg']
__supported_models__ = ['GumbelTwoStageDetector', 'MaskRCNN']


import os
import pandas as pd
from argparse import ArgumentParser
from mmdet.apis import (init_detector)
import mmcv
#import MMDet_setup.mmcv_custom  # noqa: F401,F403
import MMDet_setup.mmdet_custom  # noqa: F401,F403
import os.path as osp
from mmcv import Config, DictAction
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
import time
from MMDet_setup.global_storage.global_storage import __global_storage__

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--output_dir', default='MMDet_Bencmark_Results', help='path where to save, empty for no saving')
    args = parser.parse_args()
    return args


def main(args):
    imgs = test_imgs[0]
    WARM_UP = 250
    N_TEST = 750

    model_config = args.config
    checkpoint = args.config
    model = init_detector(model_config, checkpoint, device=args.device, cfg_options=args.cfg_options)
    model.eval()
    # ---------------------------------------------------------
    # ----- prepare the same images ready for all models ------
    # ---------------------------------------------------------
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    datas = []
    for img in imgs:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        assert False, 'CPU inference not supported for testing speed.'

    print(device)
    print(data["img"].shape)

    # ---------------------------------------------------------
    # --------- test throughput for all models ------------
    # ---------------------------------------------------------


    if "tomeact" in args.config:
        model_name = model_config.model.backbone.linkage
        reduction_ratio = model_config.model.backbone.reduction_ratio
    elif "svit" in args.config:
        model_name = "svit"
        reduction_ratio = model.config.model.backbone.keep_ratio[0]
    else:
        model_name = "ViTA"
        reduction_ratio = 1.0

    if "-t-" in args.config:
        model_name += "-T"
    else:
        model_name += "-S"


    os.makedirs(args.output_dir, exist_ok=True)
    output_filepath = os.path.join(".", args.output_dir, f"{model_name}_{reduction_ratio}_{N_TEST+WARM_UP}_{WARM_UP}.csv")

    with torch.no_grad():
        for k in range(WARM_UP):
            torch.cuda.synchronize()
        start = time.time()
        for i in range(N_TEST):
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        throughput = N_TEST / elapsed

        print(f"Throughput: {throughput:.2f} im/s")
        print('\n')

    result_dict = {"batch_size": [], "throughput": [], "reduction_ratio":[], "model":[]}
    result_dict["batch_size"].append(int(1))
    result_dict["throughput"].append(throughput)
    result_dict["reduction_ratio"].append(reduction_ratio)
    result_dict["model"].append(model_name)

    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv(output_filepath, index=False, encoding="utf-8")

if __name__ == '__main__':
    args = parse_args()
    main(args)