import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm
from math import inf
import cv2


def calculate_psnr(videos1, videos2, calculate_per_frame, calculate_final):
    print("calculate_psnr...")
    psnr = PeakSignalNoiseRatio()

    assert videos1.shape == videos2.shape

    psnr_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):

        video1 = videos1[video_num]
        video2 = videos2[video_num]

        psnr_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0)
            img2 = video2[clip_timestamp].unsqueeze(0)

            # calculate psnr of a video
            score = psnr(img1, img2)
            # print(score, 'psnr score')
            if score.item() == -inf:
                psnr_results_of_a_video.append(100)
            else:
                psnr_results_of_a_video.append(score.item())

        psnr_results.append(psnr_results_of_a_video)

    psnr_results = np.array(psnr_results)

    psnr = {}
    psnr_std = {}

    for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
        psnr[f'avg[:{clip_timestamp}]'] = np.mean(psnr_results[:,:clip_timestamp])
        psnr_std[f'std[:{clip_timestamp}]'] = np.std(psnr_results[:,:clip_timestamp])

    if calculate_final:
        psnr['final'] = np.mean(psnr_results)
        psnr_std['final'] = np.std(psnr_results)

    result = {
        "psnr": psnr,
        "psnr_std": psnr_std,
        "psnr_per_frame": calculate_per_frame,
        "psnr_video_setting": video1.shape,
        "psnr_video_setting_name": "time, channel, heigth, width",
    }

    return result
