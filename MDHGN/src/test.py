from utils import *
from ops import *


def test_network(args, network, test_loader, epoch):
    """ test function """

    """ set to eval mode """
    network.eval()

    """ Batch Iteration """
    for batch_idx, (source, realTarget, imagTarget) in enumerate(test_loader):

        """ Transfer data to GPU """
        if args.is_cuda_available:
            source = source.cuda()

        """ Run Network """
        real, imag = network(source)

        """ reduce dimension to make images """
        real = torch.squeeze(real)
        imag = torch.squeeze(imag)

        """ GPU2CPU, Torch2Numpy """
        real = real.cpu().detach().numpy()
        imag = imag.cpu().detach().numpy()

        """ save images """
        for i in range(len(image)):
            imgNum = batch_idx * len(test_loader) + i + 1
            real_fileName = combine('real', imgNum, '_epoch', epoch, '.png')
            imag_fileName = combine('imag', imgNum, '_epoch', epoch, '.png')
            real_savepath = os.path.join(
                args.savePath_of_images, real_fileName)
            imag_savepath = os.path.join(
                args.savePath_of_images, imag_fileName)
            imwrite(real[i], real_savePath)
            imwrite(imag[i], imag_savePath)


def test_time(args, network, N):
    """ test function """

    """ set to eval mode """
    network.eval()

    images = glob(os.path.join(args.path_of_dataset, 'test/*.png'))
    image = np.ndarray([N, 1, 128, 128], dtype=float)
    for i in range(N):
        image[i, 0, :, :] = imread(images[i]).astype(float)
    image = torch.from_numpy(image)
    image = image.float()

    """ Transfer data to GPU """
    if args.is_cuda_available:
        image = image.cuda()

    since = time.time()
    """ Run Network """
    _, reconimg = G(image)
    print(time.time()-since)
