import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "-m",
        "--mode",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="flag to indicate whether the test is with ldi aug or not",
    )
args = parser.parse_args()

if args.mode == "with_aug":
    for i in range(100):
        os.system('CUDA_VISIBLE_DEVICES=0 python main.py --base configs/ldi-kl-f4_test.yaml --no-test f --gpus "0,"')
else:
    list = []
    with open("used_seeds.txt", "r") as file:
        lines = file.readlines()
        for str in lines[0].split(","):
            if str != "":
                num = int(str)
                list.append(num)
    
    for seed in list:
        os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py --base configs/ldi-kl-f4_test.yaml --no-test f --seed {seed} --gpus "0,"')
    