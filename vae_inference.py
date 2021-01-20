import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
import pickle
import skimage.io
import skimage
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch.nn import functional as F
from face_models import *
from utils import *


def plot_vae(OUT_DIR):
    torch.manual_seed(999)
    latent_dim = 512

    print("Loading model...")
    model = VAE(latent_dim)
    model.load_state_dict(torch.load('save_models/face/VAE_512.pth',map_location=lambda storage, loc: storage))

    rand_variable = Variable(torch.randn(32, latent_dim), volatile=True)
    if torch.cuda.is_available():
        rand_variable = rand_variable.cuda()
        model.cuda()
    model.eval()
    rand_output = model.decode(rand_variable)
    filename = os.path.join(OUT_DIR, 'fig1_4.jpg')
    torchvision.utils.save_image(rand_output.cpu().data, filename, nrow=8)


def main(args):

    OUT_DIR = args.out_path
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    plot_vae(OUT_DIR)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW4 plot figure')
    parser.add_argument('--out_path', help='output figure directory', type=str)
    args = parser.parse_args()
    main(args)