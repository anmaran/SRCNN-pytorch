import argparse
import glob
import h5py
import numpy as np
from skimage import io, transform 
from skimage.util import img_as_uint

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for hr_path in sorted(glob.glob('{}/*'.format(args.hr_dir))):
        hr = io.imread(hr_path)
        hr_width = (hr.shape[1] // args.scale) * args.scale
        hr_height = (hr.shape[0] // args.scale) * args.scale
        hr = transform.resize(hr, output_shape=(hr_width, hr_height), order=3, preserve_range=False)
        hr = img_as_uint(hr)
        hr = np.array(hr).astype(np.float32)

        for i in range(0, hr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, hr.shape[1] - args.patch_size + 1, args.stride):
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    for lr_path in sorted(glob.glob('{}/*'.format(args.lr_dir))): 
        lr = io.imread(lr_path)
        lr_width = (lr.shape[1] // args.scale) * args.scale
        lr_height = (lr.shape[0] // args.scale) * args.scale
        lr = transform.resize(lr, output_shape=(lr_width, lr_height), order=3, preserve_range=False)
        lr = transform.resize(lr, output_shape=(lr.shape[0]* args.scale, lr.shape[1] * args.scale), order=3, preserve_range=False)
        lr = img_as_uint(lr)
        lr = np.array(lr).astype(np.float32) 

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                


    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    lr_patches = lr_patches.swapaxes(3,2).swapaxes(2,1)
    hr_patches = hr_patches.swapaxes(3,2).swapaxes(2,1)
    print(lr_patches.shape)
    print(hr_patches.shape)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, hr_path in enumerate(sorted(glob.glob('{}/*'.format(args.hr_dir)))):
        hr = io.imread(hr_path)
        hr_width = (hr.shape[1] // args.scale) * args.scale
        hr_height = (hr.shape[0]// args.scale) * args.scale
        hr = transform.resize(hr, output_shape=(hr_width, hr_height), order=3, preserve_range=False)
        hr = img_as_uint(hr)
        hr = np.array(hr).astype(np.float32)

        hr = hr.swapaxes(2,1).swapaxes(1,0)
        hr_group.create_dataset(str(i), data=hr)

    for j, lr_path in enumerate(sorted(glob.glob('{}/*'.format(args.lr_dir)))):
        lr = io.imread(lr_path)
        lr_width = (lr.shape[1] // args.scale) * args.scale
        lr_height = (lr.shape[0]// args.scale) * args.scale
        lr = transform.resize(lr, output_shape=(lr_width, lr_height), order=3, preserve_range=False)
        lr = transform.resize(lr, output_shape=(lr.shape[0] * args.scale, lr.shape[1] * args.scale), order=3, preserve_range=False)
        lr = img_as_uint(lr)
        lr = np.array(lr).astype(np.float32)

        lr = lr.swapaxes(2,1).swapaxes(1,0)
        lr_group.create_dataset(str(j), data=lr)


    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr-dir', type=str, required=True)
    parser.add_argument('--lr-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
