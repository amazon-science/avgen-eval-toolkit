import numpy as np
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
import cv2


def calculate_ms_ssim(videos1, videos2, calculate_per_frame, calculate_final):
    print("calculate Multi-Scale SSIM...")
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()

    assert videos1.shape == videos2.shape

    ms_ssim_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        ms_ssim_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0)
            img2 = video2[clip_timestamp].unsqueeze(0)

            # calculate ms_ssim of a video
            score = ms_ssim(img1, img2)
            ms_ssim_results_of_a_video.append(score.item())

        ms_ssim_results.append(ms_ssim_results_of_a_video)

    ms_ssim_results = np.array(ms_ssim_results)

    ms_ssim = {}
    ms_ssim_std = {}

    for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
        ms_ssim[f'avg[:{clip_timestamp}]'] = np.mean(ms_ssim_results[:,:clip_timestamp])
        ms_ssim_std[f'std[:{clip_timestamp}]'] = np.std(ms_ssim_results[:,:clip_timestamp])

    if calculate_final:
        ms_ssim['final'] = np.mean(ms_ssim_results)
        ms_ssim_std['final'] = np.std(ms_ssim_results)

    result = {
        "ms_ssim": ms_ssim,
        "ms_ssim_std": ms_ssim_std,
        "ms_ssim_per_frame": calculate_per_frame,
        "ms_ssim_video_setting": video1.shape,
        "ms_ssim_video_setting_name": "time, channel, heigth, width",
    }

    return result
