from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as F
import torch.nn as nn
import copy
import shutil


from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp
from tqdm import tqdm


def run_on_batch(inputs, net):
    result_batch, result_style = net(inputs.to("cuda").float(), randomize_noise=False, resize=False, return_latents=True)
    return result_batch, result_style

def run_on_batch_blender(inputs, net, net_global, target_ages):
    # this function is same as `perform_forward_pass_blender` in `coach_aging_delta.py`
    _, latent_local = net.forward(inputs.to("cuda").float(), return_latents=True)
    _, latent_global = net_global.forward(inputs.to("cuda").float(), return_latents=True)
    latent_blended = net.blender(latent_local, latent_global, target_ages=target_ages)
    result_batch, _ = net.decoder(
        [latent_blended], 
        input_is_latent=True, 
        randomize_noise=False
        )
    return result_batch, latent_blended

def helper(img_path):
    # print('Processing image: {} and saving to {}'.format(img_path, output_dir))
    # copy the image to single_img folder
    shutil.copy(img_path, output_dir_for_single_img)
    # * optional: align the image
    # try:
    #     aligned_image = run_alignment(img_path)
    # except Exception as e:
    #     print(f'Failed to align the image: {img_path}')
    #     print('Skip this image')
    #     return
    aligned_image = Image.open(img_path)
    img_transforms = EXPERIMENT_ARGS['transform']
    # print('Vision transforming done!')
    input_image = img_transforms(aligned_image)
    input_ages = img_path.split('/')[-1].split('.')[0].split('_')[0].split(' ')[0]
    # we'll run the image on multiple target ages 
    target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_transformers = [AgeTransformer(target_age=age) for age in target_ages]
    results = [np.array(aligned_image.resize((1024, 1024)))]

    for i, age_transformer in enumerate(age_transformers):
        # print(f'Running on target age: {age_transformer.target_age}')
        with torch.no_grad():
            input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
            input_image_age = torch.stack(input_image_age)

            result_tensor, result_style = run_on_batch(input_image_age, net)
            if blender:
                target_ages_blender = torch.tensor([target_ages[i] / 100]).float().to('cuda')
                input_ages_blender = torch.tensor([int(input_ages) / 100]).float().to('cuda')
                result_tensor, result_style = run_on_batch_blender(input_image_age, net, net_global, target_ages=target_ages_blender)
            result_tensor = result_tensor.squeeze(0) # (1, 3, 1024, 1024) -> (3, 1024, 1024)
            result_image = tensor2im(result_tensor)
            results.append(result_image)

            # also save result_style
            input_image_name = os.path.basename(img_path).split('.')[0]
            result_style = result_style.cpu().numpy()
            np.save(os.path.join(output_dir, f'latent_{input_image_name}_{age_transformer.target_age}.npy'), result_style)


    # save inversion results
    input_image_name = os.path.basename(img_path)
    input_img_ = np.array(aligned_image.resize((1024, 1024)))
    grid = make_grid([F.to_tensor(img) for img in [input_img_, result_image]], nrow=2)
    save_image(grid, os.path.join(output_dir, f'inversion_{input_image_name}'))
    save_image(F.to_tensor(result_image), os.path.join(output_dir_for_single_img, f'inversion_{input_image_name}'))

    # save aging results
    grid = make_grid([F.to_tensor(img) for img in results], nrow=12)
    input_image_name = os.path.basename(img_path)
    save_image(grid, os.path.join(output_dir, input_image_name))
    for i, result in enumerate(results[1:]): # skip the original image
        save_image(F.to_tensor(result), os.path.join(output_dir_for_single_img, f'{input_image_name.split(".")[0]}_{target_ages[i]}.jpg'))


def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--img_dir', type=str, help='path to the image directory')
    parser.add_argument('--desc', type=str,
                        default='', 
                        help='description of the inference')
    parser.add_argument('--model_path', type=str, help='path to the model (personalized?), default to pretrained SAM model')
    parser.add_argument('--blender', action='store_true', help='use blender')
    parser.add_argument('--output_dir', type=str, help='path to the output directory', default='/playpen-nas-ssd/luchao/projects/SAM/output_helper')
    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    img_dir = args.img_dir
    if args.desc:
        desc = args.desc
    else:
        desc_exp = args.model_path.split('/')[-4]
        desc_iter = args.model_path.split('/')[-1].split('.')[0]
        desc = f'{desc_exp}_{desc_iter}'
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, desc)
    os.makedirs(output_dir, exist_ok=True)
    output_dir_for_single_img = os.path.join(output_dir, 'single_img')
    os.makedirs(output_dir_for_single_img, exist_ok=True)

    def iterate_over_img_dir(img_dir):
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if any(file.endswith(extension) for extension in IMG_EXTENSIONS):
                    yield os.path.join(root, file)

    img_paths = sorted(list(iterate_over_img_dir(img_dir)))


    # import multiprocessing
    # print('Start processing {} images'.format(len(img_paths)))
    # with multiprocessing.Pool(4) as p:
    #     p.map(helper, [img_path for img_path in img_paths])
    # # helper(img_path)
    # print('Done!')

    EXPERIMENT_TYPE = 'ffhq_aging'
    EXPERIMENT_DATA_ARGS = {
        "ffhq_aging": {
            # default model path (pre-trained SAM model)
            "model_path": "/playpen-nas-ssd/luchao/projects/SAM/pretrained_models/sam_ffhq_aging.pt",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
    }

    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]
    model_path = EXPERIMENT_ARGS['model_path'] if args.model_path is None else args.model_path
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path

    if opts.get('exp_dir', None) is not None:
        # copy opt.json to output_dir
        opt_file = os.path.join(opts['exp_dir'], 'opt.json')
        if os.path.exists(opt_file):
            os.system('cp {} {}'.format(opt_file, output_dir))
        else:
            print("WARNING: opt.json not found in checkpoint's exp_dir")

    pprint.pprint(opts)
    opts = Namespace(**opts)
    net = pSp(opts)

    # ----------------------------- for blender only ----------------------------- #
    # blender network weights are in ckpt['blender]
    # reload encoder to get global weights
    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
    def load_global_encoder(net):
        model_path_global = EXPERIMENT_ARGS['model_path']
        print(f'Loading global encoder from {model_path_global}')
        ckpt_global = torch.load(model_path_global, map_location='cpu')
        net.encoder.load_state_dict(__get_keys(ckpt_global, 'encoder'), strict=False)
        return net
    def load_global_decoder(net):
        model_path_global = EXPERIMENT_ARGS['model_path']
        print(f'Loading global decoder from {model_path_global}')
        ckpt_global = torch.load(model_path_global, map_location='cpu')
        net.decoder.load_state_dict(__get_keys(ckpt_global, 'decoder'), strict=True)
        return net
    blender = args.blender
    if blender:
        # make a copy of the original net
        net_global = copy.deepcopy(net)
        net_global = load_global_encoder(net_global)
        net_global = load_global_decoder(net_global)
        net_global.eval()
        net_global.cuda()
        print('Global model successfully loaded using blender!')

    net.eval()
    net.cuda()


    for img_path in tqdm(img_paths):
        helper(img_path)