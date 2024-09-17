import os
import json
import pandas as pd
import torch
import argparse

from cleanfid import fid


def main(args):
    im_val_dir = args.im_val_dir
    sd_gen_base = args.sd_gen_dir


    results_dict = {"experiment": [], "duration_mean": [], "duration_std": [], "fid": []}

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    for file in sorted(os.listdir(sd_gen_base)):
        basename, ext = os.path.splitext(file)
        if ext != ".json":
            continue
            
        with open(os.path.join(sd_gen_base, file), 'r') as f:
            data = json.load(f)
        
        sd_gen_dir = data["output_dir"]    
        fid_score = fid.compute_fid(fdir1 = im_val_dir,
                                    fdir2 = sd_gen_dir,
                                    mode=args.mode,
                                    model_name="inception_v3"
                                    num_workers = 12,
                                    batch_size = 32,
                                    device=device)


        results_dict["experiment"].append(basename)
        results_dict["duration_mean"].append(data["duration_mean"])
        results_dict["duration_std"].append(data["duration_std"])
        results_dict["fid"].append(fid_score)
        print(results_dict)

    data_csv = pd.DataFrame.from_dict(results_dict)
    data_csv.to_csv(f"SD_gen_results_{args.mode}_fid.csv", sep=";", encoding='utf-8', index=False)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Computation of FID between reference ImageNet set and SD generations', add_help=False)
    parser.add_argument('--mode', default="clean", type=str)
    parser.add_argument('--im_val_dir', default="path/to/im_val_set", type=str)
    parser.add_argument('--sd_gen_dir', default="path/to/sd_generations", type=str)
    args = parser.parse_args()

    main(args)