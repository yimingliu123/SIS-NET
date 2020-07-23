import os
from glob import glob
from typing import Optional
import albumentations as albu
import torch
import yaml
from fire import Fire
from tqdm import tqdm
from matplotlib import pyplot as plt
from aug import get_normalize
from models.networks import get_poolnet

import torch

import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time

class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml') as cfg:
            config = yaml.load(cfg)
        model = get_poolnet()
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


def main(

         weights_path='godbless_fpn.h5',
         out_dir='submit/',outdir2='result/',
         ):
    def sorted_glob(pattern):
        return sorted(glob(pattern))

    # for root, _ , _ in os.walk('/home/lym/sci/testpic100/blur/'):
    for root, _, _ in os.walk('/home/lym/sci/pictest200/GROPO_dataset/blur/'):
        if root == '/home/lym/sci/pictest200/GROPO_dataset/blur/':
            continue
        print(root)
        a = root.split('/')
        print(a)
        out_dir = 'submit/'+a[-1]
        out_dir2 = '720sharp/'+a[-1]
        # imgs = sorted_glob('/home/lym/sci/testpic100/blur/' + a[-1] + '/*.png')
        # masks = sorted_glob('/home/lym/sci/testpic100/sharp/'+ a[-1]+'/*.png')##### 做第二次去模糊 只需要改动这里

        imgs = sorted_glob('/home/lym/sci/pictest200/GROPO_dataset/blur/' + a[-1] + '/*.png')  ####  更改自己的路径
        masks = sorted_glob('/home/lym/sci/pictest200/GROPO_dataset/sharp/' + a[-1] + '/*.png')  ### if '/home/lym/sci/pictest/sharp/*/*.png' is not None else [None for _ in imgs]

        pairs = zip(imgs, masks)
        # names = sorted([os.path.basename(x) for x in glob('/home/lym/sci/testpic100/blur/'+a[-1]+'/*.png')])
        names = sorted([os.path.basename(x) for x in glob('/home/lym/sci/pictest200/GROPO_dataset/blur/' + a[-1] + '/*.png')])
        predictor = Predictor(weights_path=weights_path)
        nums = len(imgs)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir2, exist_ok=True)
        time_s = time.time()
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair
            img, mask = map(cv2.imread, (f_img, f_mask))



            aug =  albu.CenterCrop(720, 720, always_apply=True)
            img = aug(image=img)["image"]
            mask = aug(image=mask)["image"]
            # plt.subplot(1,2,1)
            # plt.imshow(img)
            # plt.subplot(1,2,2)
            # plt.imshow(mask)
            # plt.show()

            col=img.shape[1]/2
            row=img.shape[0]/2
            #img =  cv2.resize(img, (int(col),int(row)), interpolation=cv2.INTER_CUBIC)
            image = predictor(img, mask)
            ####如果您想储存下来结果  此处只是测速
            #image = np.concatenate([img, image,mask], axis=1)
            #image2 = np.concatenate([image])
            cv2.imwrite(os.path.join(out_dir,name), image)
            cv2.imwrite(os.path.join(out_dir2, name), mask)



        time_k = time.time()
        ####我测试了200张  平均速度0.373

        print('Speed: %f FPS' % ((time_k-time_s)/nums))
        print('Test Done!')

if __name__ == '__main__':
    Fire(main)



