import os
import argparse
import pandas as pd
import ATC_Benchmark.models_act as models_act
from ATC_Benchmark.utils import benchmark
from timm.models import create_model


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda:0"

    model_args = argparse.Namespace(**{"reduction_ratio": args.reduction_ratio,
                "reduction_loc": [3,6,9],
                "linkage": args.linkage,
                "proportional_attn": True,
                "cluster_iters": 3,
                "equal_weight": False,
                "k_neighbors": 5
                    })

    input_size = (3,args.input_size,args.input_size)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        img_size=args.input_size,
        args = model_args
    )

    if "atc" in args.model:
        model_name = args.model + "-" + args.linkage
    else:
        model_name = args.model


    if args.use_amp:
        f16_str = "f16"
    else:
        f16_str = ""


    output_filepath = os.path.join(".", output_dir, f"{model_name}_{args.input_size}_{args.reduction_ratio[0]}_{args.runs}_{args.throw_out}_{f16_str}.csv")

    result_dict = {"batch_size": [], "throughput": [], "reduction_ratio":[], "model":[], "input_size":[]}
    for bz_k in args.batch_sizes:
        throughput = benchmark(model = model,
                            device = device,
                            input_size = input_size,
                            batch_size = int(2**bz_k),
                            runs = args.runs,
                            throw_out = args.throw_out,
                            use_fp16 = args.use_amp)

        result_dict["batch_size"].append(int(2**bz_k))
        result_dict["throughput"].append(throughput)
        result_dict["reduction_ratio"].append(args.reduction_ratio[0])
        result_dict["model"].append(model_name)
        result_dict["input_size"].append(args.input_size)

        df = pd.DataFrame.from_dict(result_dict)
        df.to_csv(output_filepath, index=False, encoding="utf-8")

    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark comparisons of the different considered Hard Merging based methods (ToMe, K-Medoids, DPC-KNN, and ATC)')
    parser.add_argument('--output_dir', default='HardMerging_Benchmark', help='path where to save, empty for no saving')
    parser.add_argument('--use_amp', action='store_true', help="")
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--runs', default=2000, type=int)
    parser.add_argument('--throw_out', default=0.25, type=float)

    parser.add_argument('--batch_sizes', type=float, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10])
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--reduction_ratio', type=float, nargs='+', default=[])
    parser.add_argument('--linkage', default="average", type=str)

    args = parser.parse_args()
    main(args)