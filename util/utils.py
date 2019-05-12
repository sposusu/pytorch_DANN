import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import params
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import os, time
from data import SynDig


def get_train_loader(dataset):
    """
    Get train dataloader of source domain or target domain
    :return: dataloader
    """
    if dataset == 'usps':
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.usps_path + '/train', transform= transform)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)

    elif dataset == 'mnistm':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.mnistm_path + '/train', transform= transform)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)

    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.svhn_path + '/train', transform= transform)


        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)
    elif dataset == 'SynDig':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = SynDig.SynDig(root= params.syndig_path, split= 'train', transform= transform, download= False)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)


    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))

    return dataloader



def get_test_loader(dataset):
    """
    Get test dataloader of source domain or target domain
    :return: dataloader
    """
    if dataset == 'usps':
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.usps_path + '/test', transform= transform)


        dataloader = DataLoader(dataset= data, batch_size= params.batch_size, shuffle= True)
    elif dataset == 'mnistm':
        transform = transforms.Compose([
            # transforms.RandomCrop((28)),
            transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.mnistm_path + '/test', transform= transform)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)
    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std = params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.svhn_path + '/test', transform= transform)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)
    elif dataset == 'SynDig':
        transform = transforms.Compose([
            transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = SynDig.SynDig(root= params.syndig_path, split= 'test', transform= transform, download= False)

        dataloader = DataLoader(dataset= data, batch_size= 1, shuffle= False)
    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))

    return dataloader



def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer



def displayImages(dataloader, length=8, imgName=None):
    """
    Randomly sample some images and display
    :param dataloader: maybe trainloader or testloader
    :param length: number of images to be displayed
    :param imgName: the name of saving image
    :return:
    """
    if params.fig_mode is None:
        return

    # randomly sample some images.
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # process images so they can be displayed.
    images = images[:length]

    images = torchvision.utils.make_grid(images).numpy()
    images = images/2 + 0.5
    images = np.transpose(images, (1, 2, 0))


    if params.fig_mode == 'display':

        plt.imshow(images)
        plt.show()

    if params.fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = 'displayImages' + str(int(time.time()))


        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')

        plt.imsave(imgName, images)
        plt.close()

    # print labels
    print(' '.join('%5s' % labels[j].item() for j in range(length)))




def plot_embedding(X, y, d, title=None, imgName=None):
    """
    Plot an embedding X with the class label y colored by the domain d.

    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param imgName: the name of saving image

    :return:
    """
    if params.fig_mode is None:
        return

    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i]/1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title(params.training_mode)
    #plt.subplot(212)
    #for i in range(X.shape[0]):
    # plot colored number
    #    plt.plot(X[i, 0], X[i, 1],
    #             color=plt.cm.bwr(y[i])
    #             )
    #plt.xticks([]), plt.yticks([])

    if params.fig_mode == 'display':
        # Directly display if no folder provided.
        plt.show()

    if params.fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = 'plot_embedding' + str(int(time.time()))

        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')

        print('Saving ' + imgName + ' ...')
        plt.savefig(imgName)
        plt.close()
    def save_checkpoint(checkpoint_path, model, optimizer):
        state = {'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}
        torch.save(state, checkpoint_path)
        print('model saved to %s' % checkpoint_path)
    
    def load_checkpoint(checkpoint_path, model, optimizer):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s' % checkpoint_path)