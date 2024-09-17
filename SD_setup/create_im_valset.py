import os
import shutil
import argparse


def main(args):
    imagenet_path = args.imagenet_path
    reduced_set_path = args.refset_path
    os.makedirs(reduced_set_path, exist_ok=True)

    for subdir, dirs, files in os.walk(imagenet_path):
        print(subdir)
        for file in sorted(files)[:5]:
            if os.path.splitext(file)[1] != ".JPEG":
                continue
            print(f"\t{file}")
            shutil.copy2(os.path.join(subdir, file), reduced_set_path)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating the ImageNet refernece set of images for FID computation')
    parser.add_argument('--imagenet_path',  help='path to the full ImageNet dataset')
    parser.add_argument('--refset_path', help="path to where the reference set of images will be save")

    args = parser.parse_args()
    main(args)
