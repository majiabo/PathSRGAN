"""
test script
"""
import sys
sys.path.append('../')
import os
from lib.dataset import wrap_image_plus
from lib import utilis
import cv2
import torch
from tqdm import tqdm
from skimage import measure
from torchvision.transforms import functional
import numpy as np

def test_psnr_ssim(target_img_path, G, data_loader):
    image_save_counter = 0
    utilis.path_checker(target_img_path)
    G.eval()
    ssims_10 = list()
    ssims_20 = list()
    psnrs_10 = list()
    psnrs_20 = list()
    for i,(img_4x,img_10x,img_20x) in tqdm(enumerate(data_loader)):
        img_4x = img_4x.cuda()
        img_10x = img_10x.cuda()
        img_20x = img_20x.cuda()
        gen10, gen20 = G(img_4x)
        for j in range(gen10.shape[0]):
            tensor_list = [img_4x[j,:,:,:],gen10[j,:,:,:],img_10x[j,:,:,:],gen20[j,:,:,:],img_20x[j,:,:,:]]
            warped_image = wrap_image_plus(tensor_list)
            # G loss situation
            image_log_name = os.path.join(target_img_path,str(image_save_counter)+'.tif')
            cv2.imwrite(image_log_name,warped_image)
            image_save_counter += 1

            # not finished yet
            gen10_ = np.array(functional.to_pil_image(gen10[j].cpu().detach()))
            img10_ = np.array(functional.to_pil_image(img_10x[j].cpu().detach()))
            gen20_ = np.array(functional.to_pil_image(gen20[j].cpu().detach()))
            img20_ = np.array(functional.to_pil_image(img_20x[j].cpu().detach()))

            ssim10 = measure.compare_ssim(img10_,gen10_,multichannel=True)
            psnr10 = measure.compare_psnr(img10_,gen10_)
            ssims_10.append(ssim10)
            psnrs_10.append(psnr10)
            ssim20 = measure.compare_ssim(img20_,gen20_,multichannel=True)
            psnr20 = measure.compare_psnr(img20_,gen20_)
            ssims_20.append(ssim20)
            psnrs_20.append(psnr20)
    avg_ssim10 = sum(ssims_10)/len(ssims_10)
    avg_ssim20 = sum(ssims_20)/len(ssims_20)
    avg_psnr10 = sum(psnrs_10)/len(psnrs_10)
    avg_psnr20 = sum(psnrs_20)/len(psnrs_20)
    return [avg_ssim10, avg_psnr10, avg_ssim20, avg_psnr20]
