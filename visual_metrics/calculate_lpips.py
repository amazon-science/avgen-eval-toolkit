import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from math import inf
import cv2


def calculate_lpips(videos1, videos2, calculate_per_frame, calculate_final, net):
    print("calculate_lpips...")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net)
    
    assert videos1.shape == videos2.shape

    lpips_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):

        video1 = videos1[video_num]
        video2 = videos2[video_num]

        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0)
            img2 = video2[clip_timestamp].unsqueeze(0)

            # calculate lpips of a video
            score = lpips(img1, img2)

            lpips_results_of_a_video.append(score.item())

        lpips_results.append(lpips_results_of_a_video)

    lpips_results = np.array(lpips_results)

    lpips = {}
    lpips_std = {}

    for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
        lpips[f'avg[:{clip_timestamp}]'] = np.mean(lpips_results[:,:clip_timestamp])
        lpips_std[f'std[:{clip_timestamp}]'] = np.std(lpips_results[:,:clip_timestamp])

    if calculate_final:
        lpips['final'] = np.mean(lpips_results)
        lpips_std['final'] = np.std(lpips_results)

    result = {
        "lpips": lpips,
        "lpips_std": lpips_std,
        "lpips_per_frame": calculate_per_frame,
        "lpips_video_setting": video1.shape,
        "lpips_video_setting_name": "time, channel, heigth, width",
    }

    return result
