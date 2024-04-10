import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from tqdm import tqdm
from math import inf
import cv2



def rgb_to_grayscale(img_tensor):
    """
    Convert an RGB image tensor to a grayscale image tensor.
    :param img_tensor: A 3xHxW tensor representing an RGB image
    :return: A 1xHxW tensor representing a grayscale image
    """
    weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    grayscale_image = torch.sum(img_tensor * weights, dim=0, keepdim=True)
    return grayscale_image

def calculate_psnrb(videos1, videos2, calculate_per_frame, calculate_final):
    print("calculate_psnrb...")
    psnrb = PeakSignalNoiseRatioWithBlockedEffect()

    assert videos1.shape == videos2.shape

    psnrb_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        psnrb_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = rgb_to_grayscale(video1[clip_timestamp]).unsqueeze(0)
            img2 = rgb_to_grayscale(video2[clip_timestamp]).unsqueeze(0)

            # calculate psnrb of a video
            score = psnrb(img1, img2)

            if score.item() == -inf:
                psnrb_results_of_a_video.append(100)
            else:
                psnrb_results_of_a_video.append(score.item())

        psnrb_results.append(psnrb_results_of_a_video)

    psnrb_results = np.array(psnrb_results)

    psnrb = {}
    psnrb_std = {}

    for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
        psnrb[f'avg[:{clip_timestamp}]'] = np.mean(psnrb_results[:,:clip_timestamp])
        psnrb_std[f'std[:{clip_timestamp}]'] = np.std(psnrb_results[:,:clip_timestamp])

    if calculate_final:
        psnrb['final'] = np.mean(psnrb_results)
        psnrb_std['final'] = np.std(psnrb_results)

    result = {
        "psnrb": psnrb,
        "psnrb_std": psnrb_std,
        "psnrb_per_frame": calculate_per_frame,
        "psnrb_video_setting": video1.shape,
        "psnrb_video_setting_name": "time, channel, heigth, width",
    }

    return result
