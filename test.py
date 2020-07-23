from glob import glob
from tqdm import tqdm
from models.networks import get_poolnet
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
import os
import cv2
import time
import torch

def main(
         dir_in = "/home/lym/生活照片/*.png",
         weights_path='godbless_fpn.h5',
         out_dir='submit/',outdir2='result/',
         ):
    def sorted_glob(pattern):
        return sorted(glob(pattern))

    def save_image(result, filename):
        save_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, save_img)

    imgs = sorted_glob(dir_in)  ####  更改自己的路径
    print(len(imgs))
    names = sorted([os.path.basename(x) for x in glob(dir_in)])
    print(names)
    nums = len(imgs)
    os.makedirs(out_dir, exist_ok=True)

    model = get_poolnet()
    model.load_state_dict(torch.load(weights_path)['model'])
    model = model.cuda()

    time_s = time.time()
    for name, image_path in tqdm(zip(names, imgs), total=len(names)):
        img_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = cv2.imread(image_path)
        # img_s = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
        # img_tensor = img_transforms(img_tensor)
        #
        # result_image = img_tensor
        # result_image = result_image.cpu().float().numpy()
        # result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
        # result_image = result_image.astype('uint8')
        # _, filename = os.path.split(image_path)
        img_s = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
        img_tensor = img_transforms(img_tensor)
        with torch.no_grad():
            img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
            result_image = model(img_tensor)
        result_image = result_image[0].cpu().float().numpy()
        result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
        result_image = result_image.astype('uint8')
        _, filename = os.path.split(image_path)

        save_image(result_image,out_dir+filename)

    time_k = time.time()
    ####我测试了200张  平均速度0.373

    print('Speed: %f FPS' % ((time_k-time_s)/nums))
    print('Test Done!')
if __name__ == '__main__':
	main("/home/lym/桌面/blur/*.png",'good.h5')