import numpy as np
import torch
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm
from math import inf
import cv2


def calculate_is(videos1, calculate_per_frame, calculate_final, layer):
    print("calculate_inception...")
    inception = InceptionScore(feature=layer, normalize=True)

    # videos [batch_size, timestamps, channel, h, w]

    inception_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        
        video1 = videos1[video_num]

        imgs1 = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0)

            imgs1.append(img1)

        imgs1 = torch.cat(imgs1, dim=0)

        inception.update(imgs1)

        score = inception.compute()

        inception_results.append(score[0].item())

    inception_results = np.array(inception_results)

    inception = {}
    inception_std = {}

    if calculate_final:
        inception['final'] = np.mean(inception_results)
        inception_std['final'] = np.std(inception_results)

    result = {
        "is": inception,
        "is_std": inception_std,
    }
    return result
