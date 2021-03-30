"""
test script
"""
import sys
sys.path.append('../')
from lib.model import PathSRGAN
import cv2
import numpy as np
import torch
from utils import utils


ckpt = './assets/ckpt/G_9_100.pth'
G = PathSRGAN(1)
G = utils.load_multi_model(G, ckpt)
G.eval()
lr_path = './assets/images/LR_001.png'
gen10_path = './assets/images/gen_MR_001.png'
gen20_path = './assets/images/gen_HR_001.png'

if __name__ == '__main__':

    # preprocess
    lr_img = cv2.imread(lr_path)[..., ::-1][None, ...]
    lr_img = lr_img.astype('float32')/255.
    lr_img = np.transpose(lr_img, axes=(0, 3, 1, 2))
    lr_img = torch.from_numpy(lr_img)
    with torch.no_grad():
        gen10, gen20 = G(lr_img)
    gen10 = np.clip(np.transpose(gen10.numpy()[0], axes=(1, 2, 0)), a_min=0, a_max=1)*255
    gen20 = np.clip(np.transpose(gen20.numpy()[0], axes=(1, 2, 0)), a_min=0, a_max=1)*255
    cv2.imwrite(gen10_path, gen10[..., ::-1].astype('uint8'))
    cv2.imwrite(gen20_path, gen20[..., ::-1].astype('uint8'))
