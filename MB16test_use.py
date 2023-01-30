import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from skimage import io, transform 
from skimage.util import img_as_uint, img_as_float32

from models import SRCNN
from utils import calc_psnr, calc_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    
    img_list = os.listdir(args.image_file)

    for img in img_list:
        img_path = os.path.join(args.image_file, img)
        print(img_path)
        image = io.imread(img_path)

        image_width = (image.shape[0] // args.scale) * args.scale
        image_height = (image.shape[1] // args.scale) * args.scale
        image = transform.resize(image, output_shape=(image_width, image_height), order=3, preserve_range=False)
        print(image.shape)
        image = transform.resize(image, output_shape=(image.shape[0] * args.scale, image.shape[1]  * args.scale), order=3, preserve_range=False)
        print(image.shape)

        image = img_as_uint(image)
        print(image.dtype)
        io.imsave(img_path.replace('.', '_bicubic_x{}.'.format(args.scale)), image, plugin=None)

        image2 = np.array(image).astype(np.float32)
        print(image2.shape)
        image2 = image2.swapaxes(2,1).swapaxes(1,0)    

        y = image2
        y /= 65535.
        print('y shp: ', y.shape)
        print('min: ', y[0,:,:].min(), 'max: ', y[0,:,:].max(), 'mean: ', y[0,:,:].mean())
        print('min: ', y[1,:,:].min(), 'max: ', y[1,:,:].max(), 'mean: ', y[1,:,:].mean())
        print('min: ', y[2,:,:].min(), 'max: ', y[2,:,:].max(), 'mean: ', y[2,:,:].mean())
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0)
        print(y.shape)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        print(y.shape)
        print(preds.shape)

        psnr = calc_psnr(y, preds)
        print('PSNR: {:.4f}'.format(psnr))

        ssim = calc_ssim(y, preds)
        print('SSIM: {:.4f}'.format(ssim))

        preds_np = preds.cpu().numpy().squeeze()
        print('min: ', preds_np[0,:,:].min(), 'max: ', preds_np[0,:,:].max(), 'mean: ', preds_np[0,:,:].mean())
        print('min: ', preds_np[1,:,:].min(), 'max: ', preds_np[1,:,:].max(), 'mean: ', preds_np[1,:,:].mean())
        print('min: ', preds_np[2,:,:].min(), 'max: ', preds_np[2,:,:].max(), 'mean: ', preds_np[2,:,:].mean())

        preds = preds.mul(65535.0).cpu().numpy().squeeze(0) 

        preds_np = np.array(preds)
        output = preds_np.swapaxes(0,1).swapaxes(1,2)

        output = np.clip(output, 0.0, 65535.0).astype(np.uint16)
        print(output.dtype)


        io.imsave(img_path.replace('.', '_srcnn_x{}.'.format(args.scale)), output, plugin=None)
