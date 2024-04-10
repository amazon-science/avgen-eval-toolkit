import numpy as np
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
from math import inf
import cv2


def calculate_clip(videos1, caps, calculate_per_frame, calculate_final, clip_model):
    print("calculate_CLIP Score...")

    metric = CLIPScore(model_name_or_path=clip_model)

    # videos [batch_size, timestamps, channel, h, w]

    clip_results = []
    idx = 0
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        text  = caps[video_num]
        video1 = videos1[video_num]

        clip_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp] * 255

            # calculate clip of a video
            score = metric(img1, text)
            print(score)

            clip_results_of_a_video.append(score.item())


        clip_results.append(clip_results_of_a_video)
        

    clip_results = np.array(clip_results)

    clip = {}
    clip_std = {}

    for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
        clip[f'avg[:{clip_timestamp}]'] = np.mean(clip_results[:,:clip_timestamp])
        clip_std[f'std[:{clip_timestamp}]'] = np.std(clip_results[:,:clip_timestamp])

    if calculate_final:
        clip['final'] = np.mean(clip_results)
        clip_std['final'] = np.std(clip_results)

    result = {
        "clip": clip,
        "clip_std": clip_std,
        "clip_per_frame": calculate_per_frame,
        "clip_video_setting": video1.shape,
        "clip_video_setting_name": "time, channel, heigth, width",
    }

    return result