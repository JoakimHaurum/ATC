import os
import argparse
import pandas as pd
from ATC_Benchmark.utils import benchmark


def get_framework_model(framework, num_clusters, linkage):
    if framework == "sklearn":
        from ATC_Benchmark.sklearn_atc import Block_ATC
    elif framework == "scipy":
        from ATC_Benchmark.scipy_atc import Block_ATC
    elif framework == "rapids":
        from torch.cuda.memory import change_current_allocator
        from rmm.allocators.torch import rmm_torch_allocator
        change_current_allocator(rmm_torch_allocator)           
        from ATC_Benchmark.rapids_atc import Block_ATC
    
    return Block_ATC(num_clusters=num_clusters, linkage=linkage)

def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda:0"
    model_name = args.framework + "_" + args.linkage

    if args.use_amp:
        f16_str = "f16"
    else:
        f16_str = ""


    output_filepath = os.path.join(".", output_dir, f"{model_name}_{args.emb_size}_{args.reduction_ratio[0]}_{args.runs}_{args.throw_out}_{f16_str}.csv")
    result_dict = {"batch_size": [], "throughput": [], "reduction_ratio":[], "model":[], "img_size":[], "n_patches":[], "emb_size":[]}
    model = None
    
    for is_k in args.input_sizes:
        n_patches = int(is_k**2/16**2)
        num_clusters = int(n_patches*args.reduction_ratio[0])
        del model
        model = get_framework_model(args.framework, num_clusters, args.linkage)
        input_size = (n_patches,args.emb_size)
        
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
            result_dict["img_size"].append(is_k)
            result_dict["n_patches"].append(n_patches)
            result_dict["emb_size"].append(args.emb_size)

            df = pd.DataFrame.from_dict(result_dict)
            df.to_csv(output_filepath, index=False, encoding="utf-8")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='ATC Benchmark across frameworks - Only considers the clustering module')
    parser.add_argument('--output_dir', default='ATC_Benchmark_IM_Size_Results', help='path where to save, empty for no saving')
    parser.add_argument('--use_amp', action='store_true', help="")
    parser.add_argument('--input_sizes', nargs='+', type=int, default=[224, 256, 384, 512, 1024, 2048])
    parser.add_argument('--emb_size', default=384, type=int)
    parser.add_argument('--runs', default=1000, type=int)
    parser.add_argument('--throw_out', default=0.25, type=float)

    parser.add_argument('--batch_sizes', type=float, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10])
    parser.add_argument('--framework', default='sklearn', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--reduction_ratio', type=float, nargs='+', default=[])
    parser.add_argument('--linkage', default="average", type=str)
    args = parser.parse_args()

    main(args)