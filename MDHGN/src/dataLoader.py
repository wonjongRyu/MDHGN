from os.path import join
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class myDataset(Dataset):
    def __init__(self, args, phase):
        self.source = glob(join(args.path_of_dataset, phase, "Source/*.*"))
        self.real = glob(join(args.path_of_dataset, phase, "RH/*.*"))
        self.imag = glob(join(args.path_of_dataset, phase, "IH/*.*"))
        self.sz = args.size_of_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        source = np.asarray(imread(self.source[idx]))
        source = np.reshape(source, (self.sz, self.sz, 1))
        source = np.swapaxes(source, 0, 2)
        source = np.swapaxes(source, 1, 2)

        real = np.asarray(imread(self.real[idx]))
        real = np.reshape(real, (self.sz, self.sz, 1))
        real = np.swapaxes(real, 0, 2)
        real = np.swapaxes(real, 1, 2)

        imag = np.asarray(imread(self.imag[idx]))
        imag = np.reshape(imag, (self.sz, self.sz, 1))
        imag = np.swapaxes(imag, 0, 2)
        imag = np.swapaxes(imag, 1, 2)

        return source, real, imag


def data_loader(args):
    """ Data Loader"""

    """ Load image data """
    train_images = myDataset(args, "train")
    valid_images = myDataset(args, "valid")
    test_images = myDataset(args, "test")

    """ Wrap them with DataLoader structure """
    train_loader = DataLoader(
        train_images, batch_size=args.size_of_miniBatches, shuffle=True)
    valid_loader = DataLoader(
        valid_images, batch_size=args.size_of_miniBatches, shuffle=True)
    test_loader = DataLoader(
        test_images, batch_size=args.size_of_miniBatches, shuffle=False)  # ***FALSE***

    return train_loader, valid_loader, test_loader
