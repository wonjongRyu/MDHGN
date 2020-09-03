import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from ops import *
import time
from test import test, test_refinement
from data import data_loader
import pytorch_ssim
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
        if (epoch % args.saveCycle_of_images) == 0:
            # visualize_conv_layer(args, G)
            test(args, G, test_loader, epoch)

        """ Save model """
        if (epoch % args.saveCycle_of_models) == 0:
            torch.save(G.state_dict(), args.savePath_of_models +
                       "/HGN_train_continued" + str(epoch) + ".pt")

    print('======================[ train finished ]======================')


def iteratation(args, network, data_loader, phase):
    """ iteration function """

    """ Phase setting: train or valid """
    if phase == "train":
        network.train()
    if phase == "valid":
        network.eval()

    """ Define loss function and optimizer """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(G.parameters(), lr=args.learning_rate)

    """ Initialize the loss_sum """
    loss_mse_sum = 0.0
    loss_ssim_sum = 0.0

    """ Start batch iteration """
    for batch_idx, (source, realTarget, imagTarget) in enumerate(data_loader):

        dataNum = len(data_loader)

        """ Transfer data to GPU """
        if args.is_cuda_available:
            source, realTarget, imagTarget = source.cuda(), realTarget.cuda(), imagTarget.cuda()

        """ Run model """
        real, imag = G(source)

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
