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



def plot_gan(OUT_DIR):
    torch.manual_seed(20927)
    
    #fig2_3
    rand_inputs = Variable(torch.randn(32, 100, 1, 1),volatile=True)
    G = Generator()
    G.load_state_dict(torch.load('save_models/face/Generator.pth',map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        rand_inputs = rand_inputs.cuda()
        G.cuda()
    G.eval()
    rand_outputs = G(rand_inputs)
    filename = os.path.join(OUT_DIR, 'fig2_2.jpg')
    torchvision.utils.save_image(rand_outputs.cpu().data, filename, nrow=8)

def main(args):

    OUT_DIR = args.out_path
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    plot_gan(OUT_DIR)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW4 plot figure')
    parser.add_argument('--out_path', help='output figure directory', type=str)
    args = parser.parse_args()
    main(args)