from utils import *
from ops import *


def test_network(args, network, test_loader, epoch):
    """ test function """

    """ set to eval mode """
    network.eval()

    """ Batch Iteration """
    for batch_idx, (source, realTarget, imagTarget) in enumerate(test_loader):

        dataNum = len(source)

        """ Transfer data to GPU """
        if args.is_cuda_available:
            source = source.cuda()

        """ Run Network """
        real, imag = network(source)

        """ reduce dimension to make images """
        real = torch.squeeze(real)
        imag = torch.squeeze(imag)
        source = torch.squeeze(source)

        """ GPU2CPU, Torch2Numpy """
        real = real.cpu().detach().numpy()
        imag = imag.cpu().detach().numpy()
        source = source.cpu().detach().numpy()

        """ save images """
        for i in range(dataNum):
            imgNum = batch_idx * dataNum + i + 1
            source_fileName = combine(
                'epoch', epoch, '_img', imgNum, '_source', '.png')
            real_fileName = combine(
                'epoch', epoch, '_img', imgNum, '_real', '.png')
            imag_fileName = combine(
                'epoch', epoch, '_img', imgNum, '_imag', '.png')
            source_savePath = os.path.join(
                args.savePath_of_images, source_fileName)
            real_savePath = os.path.join(
                args.savePath_of_images, real_fileName)
            imag_savePath = os.path.join(
                args.savePath_of_images, imag_fileName)
            imwrite(source[i], source_savePath)
            imwrite(real[i], real_savePath)
            imwrite(imag[i], imag_savePath)
