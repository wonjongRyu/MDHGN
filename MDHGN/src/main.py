# External modules

import argparse

# From this package

from MDHGN import MDHGN
from train import train_network
from data import *
from utils import *
from test import test_network, test_time


def parse_args():
    """ parser_args function """
    parser = argparse.ArgumentParser()

    """ Description """
    parser.add_argument("--description", type=str, default="vanilla version")
    parser.add_argument("--start_time", type=str,
                        default="Year/Month/Day_Hour/Minutes")

    """ GPU """
    parser.add_argument("--is_cuda_available", type=int, default=True)
    parser.add_argument("--device_number", type=int, default=0)

    """ Pretrained Network """
    parser.add_argument("--use_pretrained_network", type=int, default=False)
    parser.add_argument("--date_of_pretrained_network_date",
                        type=str, default="200314_0912")
    parser.add_argument("--name_of_pretrained_network", type=str,
                        default="HGN_train_continued6.pt")

    """ Dataset """
    parser.add_argument("--path_of_dataset", type=str,
                        default="../dataset")
    parser.add_argument("--size_of_images", type=int, default=512)

    """ Hyperparameters: Architecture """
    parser.add_argument("--size_of_miniBatches", type=int, default=16)
    parser.add_argument("--num_of_initChannel", type=int, default=128)
    parser.add_argument("--num_of_blocks", type=int, default=15)
    parser.add_argument("--num_of_epochs", type=int, default=1000)

    """ Hyperparameters: Learning Rate """
    parser.add_argument("--learning_rate", type=float, default=10e-4)
    parser.add_argument("--coefficient_of_loss", type=float, default=0.05)

    """ Save Cycles """
    parser.add_argument("--saveCycle_of_loss", type=int, default=1)
    parser.add_argument("--saveCycle_of_image", type=int, default=1)
    parser.add_argument("--saveCycle_of_network", type=int, default=3)

    """ Save Paths """
    parser.add_argument("--savePath_of_outputs",
                        type=str, default="../outputs")
    parser.add_argument("--savePath_of_images", type=str,
                        default="../outputs/images")
    parser.add_argument("--savePath_of_networks", type=str,
                        default="../outputs/networks")
    parser.add_argument("--savePath_of_layers", type=str,
                        default="../outputs/layers")
    parser.add_argument("--savePath_of_tensor", type=str,
                        default="../outputs/tensor")
    parser.add_argument("--savePath_of_loss", type=str,
                        default="../outputs/loss.csv")
    parser.add_argument("--savePath_of_arch", type=str,
                        default="../outputs/arch.csv")
    parser.add_argument("--savePath_of_args", type=str,
                        default="../outputs/args.csv")
    parser.add_argument("--savePath_of_test", type=str,
                        default="../outputs/test.csv")

    return parser.parse_args()


def main():
    """ Load Arguments """
    args = parse_args()
    args.description = "Pytorch implementation of Multi-depth Hologram Generation Network (MDHGN)"

    """ Define Network """
    network = MDHGN(args)

    """ Load Pretrained Network """
    if args.use_pretrained_network:
        loadPath_of_pretrained_network = os.path.join(
            "../outputs", args.start_time, "networks", args.network_name)
        network.load_state_dict(torch.load(loadPath_of_pretrained_network))

    """ CPU2GPU """
    if args.is_cuda_available:
        torch.cuda.set_device(args.device_number)
        network.cuda()

    """ initialization """
    write_start_time(args)
    make_csvfile_and_folders(args)
    summary_architecture_of_network_(
        args, network, (1, args.size_of_images, args.size_of_images))

    """ Train Network """
    train_network(args, network)


if __name__ == "__main__":
    main()
