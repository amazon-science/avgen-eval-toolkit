import os
import librosa
import torch
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool


class AudExtractor:
    def __init__(self, *args, **kwargs):
        self.aud_path_list = kwargs.get("aud_paths", args[0])
        self.nj = kwargs.get("nj", 24)
    def extract(self):
        print("Extracting audio files")
        aud_feats = []
        for aud_loc in tqdm(self.aud_path_list):
            data = (np.load(aud_loc))
            aud_feats.append(np.array(data)) #(seq_len, 128)
        return aud_feats

class VidExtractor:
    def __init__(self, *args, **kwargs):
        self.vid_path_list = kwargs.get("aud_paths", args[0])
        self.nj = kwargs.get("nj", 24)
    def extract(self):
        print("Extracting visual files")
        vid_feats = []
        for vid_loc in tqdm(self.vid_path_list):
            data = (np.load(vid_loc))
            vid_feats.append(np.array(data)) #(seq_len, 1024)
        return vid_feats

def unpack_torch_segment(padded_segment, duration):
    batch_num = padded_segment.size(0)
    result = []
    for idx in range(batch_num):
        cur_segment = padded_segment[idx]
        
        cur_dur = duration[idx]
        cut_seg = cur_segment[:cur_dur]
        result.append(cut_seg)
    resutl = torch.Tensor(result)
    return result

