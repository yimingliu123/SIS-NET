from __future__ import print_function
import argparse
import numpy as np
import torch
import cv2
import yaml
import os
from torchvision import models, transforms
from torch.autograd import Variable
import shutil
import glob
import tqdm
from util.metrics import PSNR
from albumentations import Compose, CenterCrop, PadIfNeeded
from PIL import Image
# from ssim.ssimlib import SSIM
# from util.metrics import SSIM
# from models.networks import get_generator
import time
from models.networks import get_poolnet

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def get_args():
	parser = argparse.ArgumentParser('Test an image')
	# parser.add_argument('--img_folder', required=True, help='GoPRO Folder')
	# parser.add_argument('--weights_path', required=True, help='Weights path',default="best_fpn.h5")

	return parser.parse_args()


def prepare_dirs(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


def get_gt_image(path):
	dir, filename = os.path.split(path)
	base, seq = os.path.split(dir)
	base, _ = os.path.split(base)
	img = cv2.cvtColor(cv2.imread(os.path.join(base, 'sharp', seq, filename)), cv2.COLOR_BGR2RGB)
	return img

def save_image(result,filename):
	save_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
	cv2.imwrite(filename,save_img)


def test_image(model, image_path):
	img_transforms = transforms.Compose([
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	# size_transform = Compose([
	# 	PadIfNeeded(736, 1280)
	# ])
	#crop = CenterCrop(720, 720)
	img = cv2.imread(image_path)
	img_s = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# img_s = size_transform(image=img)['image']
	#img_s = crop(image=img)['image']
	img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
	img_tensor = img_transforms(img_tensor)
	with torch.no_grad():
		img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
		result_image = model(img_tensor)
	result_image = result_image[0].cpu().float().numpy()
	result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
	result_image = result_image.astype('uint8')
	# gt_image = get_gt_image(image_path)
	_, filename = os.path.split(image_path)
	#save_image(result_image,filename)
	psnr = PSNR(result_image, img_s)
	pilFake = Image.fromarray(result_image)
	pilReal = Image.fromarray(img_s)
	ssim = calculate_ssim(result_image, img_s)
	return psnr, ssim


def test(model, files):
	psnr = 0
	ssim = 0
	for file in tqdm.tqdm(files):
		cur_psnr, cur_ssim = test_image(model, file)
		psnr += cur_psnr
		ssim += cur_ssim
	print("PSNR = {}".format(psnr / len(files)))
	print("SSIM = {}".format(ssim / len(files)))


if __name__ == '__main__':
	weights_path = "good.h5"
	args = get_args()
	with open('config/config.yaml') as cfg:
		config = yaml.load(cfg)
	model = get_poolnet()
	model.load_state_dict(torch.load(weights_path)['model'])
	model = model.cuda()
	begin = time.time()
	filenames = sorted(glob.glob("/home/lym/下载/DeblurpoolGANv2-master/GOPRO/GOPRO_3840FPS_AVG_3-21/test/blur/*/*.png", recursive=True))
	print(len(filenames))
	test(model, filenames)
	final = time.time()
	print("time is " , final-begin)
