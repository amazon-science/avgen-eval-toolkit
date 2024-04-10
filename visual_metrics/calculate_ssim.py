import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
import cv2


def calculate_ssim(videos1, videos2, calculate_per_frame, calculate_final):
    print("calculate_ssim...")
    ssim = StructuralSimilarityIndexMeasure()

    assert videos1.shape == videos2.shape

    ssim_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        ssim_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0)
            img2 = video2[clip_timestamp].unsqueeze(0)

            # calculate ssim of a video
            score = ssim(img1, img2)
            ssim_results_of_a_video.append(score.item())

        ssim_results.append(ssim_results_of_a_video)

    ssim_results = np.array(ssim_results)

    ssim = {}
    ssim_std = {}

    for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
        ssim[f'avg[:{clip_timestamp}]'] = np.mean(ssim_results[:,:clip_timestamp])
        ssim_std[f'std[:{clip_timestamp}]'] = np.std(ssim_results[:,:clip_timestamp])

    if calculate_final:
        ssim['final'] = np.mean(ssim_results)
        ssim_std['final'] = np.std(ssim_results)

    result = {
        "ssim": ssim,
        "ssim_std": ssim_std,
        "ssim_per_frame": calculate_per_frame,
        "ssim_video_setting": video1.shape,
        "ssim_video_setting_name": "time, channel, heigth, width",
    }

    return result
