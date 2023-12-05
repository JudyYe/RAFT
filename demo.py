from tqdm import tqdm
import os.path as osp
import sys
sys.path.append('core')
import torch.nn.functional as F
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


from jutils import image_utils

DEVICE = 'cuda'

def warp_image(img, flo):
    """

    Args:
        img (_type_): image of shape [B,C,H,W]
        flo (_type_): flow of shape [B,2,H,W]
    """
    B, C, H, W = img.shape
    x = torch.arange(W).view(1, -1).expand(H, -1).float().to(DEVICE)
    y = torch.arange(H).view(-1, 1).expand(-1, W).float().to(DEVICE)
    grid = torch.stack([x, y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid + flo
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0
    grid = grid.permute(0, 2, 3, 1)
    out = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return out


def warp_image_np(curImg, flow):
    """

    Args:
        img (_type_): (H, W, 3)
        flo (_type_): (H, W, 2)
    """
    h, w = flow.shape[:2]
    # flow = -flow
    flow_out = flow.copy()
    flow_out[:,:,0] += np.arange(w)  # x,y 
    flow_out[:,:,1] += np.arange(h)[:,np.newaxis]  # 0, ... N-1
    prevImg = cv2.remap(curImg, flow_out, None, cv2.INTER_LINEAR)    
    return prevImg


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, img2, flo, save_path=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    # print(flo)
    # flow: 1. uv in pixel space
    img2_pred = warp_image_np(img, flo)
    img1_pred = warp_image_np(img2, flo)
    out = np.concatenate([img2_pred - img2, img1_pred - img], 0)
    cv2.imwrite(save_path[:-4] + '_wrap.png', out[..., ::-1])

    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)


    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    cv2.imwrite(save_path, img_flo[:, :, [2,1,0]])


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in tqdm(zip(images[:-1], images[1:]), total=len(images)-1):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            index = osp.basename(imfile1)[:-4]

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # Flow is warp from 2 to 1
            image1_pred = warp_image(image2, flow_up)
            image_utils.save_images( 
                torch.cat([image1, image1 - image2,
                image1 - image1_pred], -2) / 255, osp.join(save_dir,  f'{index}_1to2'))

            image2_pred = warp_image(image1, -flow_up)
            image_utils.save_images((image2 - image2_pred)/255, osp.join(save_dir, f'{index}_2to1'))
            print('flow shape', flow_up.shape, image1.shape)

            viz(image1, image2, flow_up, osp.join(save_dir, osp.basename(imfile1)[:-4] + '.png'))
            save_flow(flow_up, osp.join(save_dir, osp.basename(imfile1)[:-4] + '.npz'))


def save_flow(flow, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    flow = flow[0].permute(1,2,0).cpu().numpy()
    np.savez_compressed(save_path, flow=flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--save_dir', help="save_dir", default='/private/home/yufeiy2/scratch/result/vis_raft')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    save_dir = args.save_dir
    demo(args)
