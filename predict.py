"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim

import numpy as np

from models import models
from train import  params
from util import utils
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
import argparse, sys, os
import csv


import torch
from torch.autograd import Variable

import time

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def test(feature_extractor, class_classifier, domain_classifier, target_dataloader):
    """
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :return: None
    """
    # setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()
    source_correct = 0.0
    target_correct = 0.0
    domain_correct = 0.0
    tgt_correct = 0.0
    src_correct = 0.0
    file_fn = []
    ans = []


    for batch_idx, tdata in enumerate(target_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(target_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        input2, label2 = tdata
        if params.use_gpu:
            input2= Variable(input2.cuda())
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        else:
            input2= Variable(input2)
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

        output2 = class_classifier(feature_extractor(input2))
        pred2 = output2.data.max(1, keepdim=True)[1]
        ans.extend(pred2.data[0])
        file_fn.extend(label2)
        break
    return ans, file_fn



   

def main(args):

    # Set global parameters.
    params.fig_mode = args.fig_mode
    params.epochs = args.max_epoch
    params.training_mode = args.training_mode
    params.source_domain = args.source_domain
    params.target_domain = args.target_domain
    if params.embed_plot_epoch is None:
        params.embed_plot_epoch = args.embed_plot_epoch
    params.lr = args.lr


    if args.save_dir is not None:
        params.save_dir = args.save_dir
    else:
        print('Figures will be saved in ./experiment folder.')

    # prepare the source data and target data
    tgt_test_dataloader = utils.get_pred_loader(params.target_domain,args.test_dir)

    if params.fig_mode is not None:
        print('Images from training on source domain:')

        # utils.displayImages(src_train_dataloader, imgName='source')

        print('Images from test on target domain:')
        # utils.displayImages(tgt_test_dataloader, imgName='target')

    # init models
    #model_index = params.source_domain + '_' + params.target_domain
    feature_extractor = models.SVHN_Extractor()#params.extractor_dict[model_index]
    class_classifier =models.SVHN_Class_classifier() #params.class_dict[model_index]
    domain_classifier = models.SVHN_Domain_classifier()#params.domain_dict[model_index]

    load_checkpoint(params.extractor_dict[params.target_domain],feature_extractor)
    load_checkpoint(params.class_dict[params.target_domain],class_classifier)
    load_checkpoint(params.domain_dict[params.target_domain],domain_classifier)

    if params.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()

    # init criterions
    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    # init optimizer
    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr= params.lr, momentum= 0.9)

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))

        ans, name = test(feature_extractor, class_classifier, domain_classifier, tgt_test_dataloader)

    with open(params.save_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name','label'])

        for a, n in zip(ans, name):
            writer.writerow([n,a])



def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type= str, default= 'MNIST', help= 'Choose source domain.')

    parser.add_argument('--target_domain', type= str, default= 'MNIST_M', help = 'Choose target domain.')

    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')

    parser.add_argument('--test_dir', type=str, default=None, help='test dor')

    parser.add_argument('--save_dir', type=str, default=None, help='Path to save plotted images.')

    parser.add_argument('--training_mode', type=str, default='dann', help='Choose a mode to train the model.')

    parser.add_argument('--max_epoch', type=int, default=100, help='The max number of epochs.')

    parser.add_argument('--embed_plot_epoch', type= int, default=5, help= 'Epoch number of plotting embeddings.')

    parser.add_argument('--lr', type= float, default= 0.01, help= 'Learning rate.')

    return parser.parse_args()



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
