import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from ops import *
import time
from test import test_network
from dataLoader import data_loader
from torchvision import transforms


def train_network(args, network):
    """ train function """

    since = time.time()

    """ Start iteration """
    for epoch in range(1, args.num_of_epochs+1):

        """ run 1 epoch and get loss """
        train_loader, valid_loader, test_loader = data_loader(args)
        train_loss = iteration(args, network, train_loader, phase="train")
        valid_loss = iteration(args, network, valid_loader, phase="valid")

        """ Print loss """
        if (epoch % args.saveCycle_of_loss) == 0:
            print_loss(epoch, time.time()-since, train_loss, valid_loss)
            record_on_csv(
                args, epoch, time.time()-since, train_loss, valid_loss)

        """ Print image """
        if (epoch % args.saveCycle_of_image) == 0:
            test_network(args, network, test_loader, epoch)

    print('======================[ train finished ]======================')


def iteration(args, network, data_loader, phase):
    """ iteration function """

    """ Phase setting: train or valid """
    if phase == "train":
        network.train()
    if phase == "valid":
        network.eval()

    """ Define loss function and optimizer """
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)

    """ Initialize the loss_sum """
    loss_sum = 0
    MSE_real_sum = 0
    MAE_real_sum = 0
    MSE_imag_sum = 0
    MAE_imag_sum = 0

    """ Start batch iteration """
    for batch_idx, (source, realTarget, imagTarget) in enumerate(data_loader):

        dataNum = len(source)

        """ Transfer data to GPU """
        if args.is_cuda_available:
            source, realTarget, imagTarget = source.cuda(), realTarget.cuda(), imagTarget.cuda()

        """ Run Network """
        real, imag = network(source)

        """ Calculate batch loss """
        MSE_real = MSE(real, realTarget)
        MAE_real = MAE(real, realTarget)
        MSE_imag = MSE(imag, imagTarget)
        MAE_imag = MAE(imag, imagTarget)

        loss = (MSE_real+MSE_imag) + 0.1*(MAE_real+MAE_imag)

        """ Back propagation """
        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """ Add to get epoch loss """
        loss_sum += loss.item()  # 여기 item() 없으면 GPU 박살
        MSE_real_sum += MSE_real.item()
        MAE_real_sum += MAE_real.item()
        MSE_imag_sum += MSE_imag.item()
        MAE_imag_sum += MAE_imag.item()

        """ Clear memory """
        torch.cuda.empty_cache()

    return loss_sum/dataNum, MSE_real_sum/dataNum, MAE_real_sum/dataNum, MSE_imag_sum/dataNum, MAE_imag_sum/dataNum
