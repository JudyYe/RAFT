# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

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

from demo import warp_image, load_image
from jutils import image_utils

DEVICE = 'cuda'
save_dir = '/private/home/yufeiy2/scratch/result/vis_raft'


def infer_flow_2_crops(model, bboxes, image1, image2, resize=224):
    """

    Args:
        model (_type_): flow model
        bboxes (_type_): (N, 2, 4) in xyxy format
        img1 (_type_): (N, C, H, W)
        img2 (_type_): (N, C, H, W)
    """
    # (N, 2, H, W)
    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    
    # transform by crop xy
    x1_prev, y1_prev = bboxes[:, 0, 0], bboxes[:, 0, 1]  # (N, 4)
    x1_curr, y1_curr = bboxes[:, 1, 0], bboxes[:, 1, 1]  # (N, 4)
    x2_prev, y2_prev = bboxes[:, 0, 2], bboxes[:, 0, 3]  # (N, 4)
    x2_curr, y2_curr = bboxes[:, 1, 2], bboxes[:, 1, 3]  # (N, 4)

    flow_up[:, 0, :, :] += x1_prev[:, None, None]
    flow_up[:, 1, :, :] += y1_prev[:, None, None]
    flow_up[:, 0, :, :] -= x1_curr[:, None, None]
    flow_up[:, 1, :, :] -= y1_curr[:, None, None]

    # transform by resize the crops to resize shape? 
    dx_prev, dy_prev = x2_prev - x1_prev, y2_prev - y1_prev
    dx_curr, dy_curr = x2_curr - x1_curr, y2_curr - y1_curr

    flow_up[:, 0, :, :] *= resize / dx_prev[:, None, None]
    flow_up[:, 1, :, :] *= resize / dy_prev[:, None, None]
    flow_up[:, 0, :, :] *= dx_curr[:, None, None] / resize
    flow_up[:, 1, :, :] *= dy_curr[:, None, None] / resize
    
    # crop the flow
    x1_int, y1_int = x1_curr.long(), y1_curr.long()
    x2_int, y2_int = x2_curr.long(), y2_curr.long()
    flow_list = []
    for i in range(flow_up.shape[0]):
        flow = flow_up[i, :, y1_int[i]:y2_int[i], x1_int[i]:x2_int[i]]
        flow = F.interpolate(flow[None], size=(resize, resize), mode='bilinear', align_corners=False)[0]
        flow_list.append(flow)
    flow_up = torch.stack(flow_list, dim=0)
    return flow_up


def load_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model

def demo_crop(args):
    model = load_model(args)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            H = 224
            bboxes = np.array([[10, 10, 224, 224], [24, 24, 256, 256]])  # (2, 4)
            bboxes = np.tile(bboxes[None], (image1.shape[0], 1, 1))  # (N, 2, 4)

            crop1 = image_utils.crop_resize(np.array(Image.open(imfile1)).astype(np.uint8), bboxes[0, 0], H)
            crop2 = image_utils.crop_resize(np.array(Image.open(imfile2)).astype(np.uint8), bboxes[0, 1], H)
            crop1 = torch.from_numpy(crop1).permute(2, 0, 1).float()[None].to(DEVICE)
            crop2 = torch.from_numpy(crop2).permute(2, 0, 1).float()[None].to(DEVICE)  

            index = osp.basename(imfile1)[:-4]

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            bboxes = torch.from_numpy(bboxes).float().to(DEVICE)

            flow_up = infer_flow_2_crops(model, bboxes, image1, image2, resize=H)
            # Flow is warp from 2 to 1
            crop1_pred = warp_image(crop2, flow_up)
            image_utils.save_images(torch.cat(
                [crop1_pred-crop1, crop1 - crop2], -2)/255, osp.join(save_dir, index + '_crop'))
            
            

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo_crop(args)