import numpy as np
import torch
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from tqdm import tqdm
from math import inf
import cv2


def calculate_mifid(videos1, videos2, calculate_per_frame, calculate_final, layer):
    print("calculate_mifid...")
    mifid = MemorizationInformedFrechetInceptionDistance(feature=layer)

    assert videos1.shape == videos2.shape

    mifid_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        imgs1, imgs2 = [], []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp] * 255
            img2 = video2[clip_timestamp] * 255

            # calculate mifid of a video
            img1, img2 = np.array(img1), np.array(img2)
            # Convert numpy array to torch tensor with dtype uint8
            img1 = torch.from_numpy(img1).type(torch.uint8)
            img2 = torch.from_numpy(img2).type(torch.uint8)

            img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)

            imgs1.append(img1)
            imgs2.append(img2)

        imgs1 = torch.cat(imgs1, dim=0)
        imgs2 = torch.cat(imgs2, dim=0)

        mifid.update(imgs1, real=True)
        mifid.update(imgs2, real=False)

        score = mifid.compute()

        mifid_results.append(score.item())

    mifid_results = np.array(mifid_results)

    mifid = {}
    mifid_std = {}

    if calculate_final:
        mifid['final'] = np.mean(mifid_results)
        mifid_std['final'] = np.std(mifid_results)

    result = {
        "mifid": mifid,
        "mifid_std": mifid_std,
    }
    return result
