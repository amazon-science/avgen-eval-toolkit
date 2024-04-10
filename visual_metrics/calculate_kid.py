import numpy as np
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
from math import inf
import cv2


def calculate_kid(videos1, videos2, calculate_per_frame, calculate_final, layer, s_size):
    print("calculate_kid...")
    kid = KernelInceptionDistance(subset_size=s_size, feature=layer)

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    kid_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        imgs1, imgs2 = [], []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp] * 255
            img2 = video2[clip_timestamp] * 255

            # calculate kid of a video
            img1, img2 = np.array(img1), np.array(img2)
            # Convert numpy array to torch tensor with dtype uint8
            img1 = torch.from_numpy(img1).type(torch.uint8)
            img2 = torch.from_numpy(img2).type(torch.uint8)

            img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)

            imgs1.append(img1)
            imgs2.append(img2)

        imgs1 = torch.cat(imgs1, dim=0)
        imgs2 = torch.cat(imgs2, dim=0)

        kid.update(imgs1, real=True)
        kid.update(imgs2, real=False)

        kid_mean, _ = kid.compute()

        kid_results.append(kid_mean.item())

    kid_results = np.array(kid_results)

    kid = {}
    kid_std = {}

    if calculate_final:
        kid['final'] = np.mean(kid_results)
        kid_std['final'] = np.std(kid_results)

    result = {
        "kid": kid,
        "kid_std": kid_std,
    }
    return result
