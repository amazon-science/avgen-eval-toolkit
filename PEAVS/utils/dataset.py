import torch
import torch.nn as nn
import torch.utils as torch_utils
import numpy as np
import sys


class AudVidSet(torch_utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super(AudVidSet, self).__init__()
        self.aud_list = kwargs.get("aud_list", args[0]) # (N, D, T)
        self.vid_list = kwargs.get("vid_list", args[1])
        
        # check max duration
        self.max_dur = np.min([np.max([len(cur_aud) for cur_aud in self.aud_list])])

    def __len__(self):
        return len(self.aud_list)

    def __getitem__(self, idx):
        cur_aud = self.aud_list[idx][:self.max_dur]
        cur_vid = self.vid_list[idx]

        result = (cur_aud, cur_vid)
        return result

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    total_aud = []
    total_vid = []
    for aud, vid, in batch:
        total_aud.append(torch.Tensor(aud))
        total_vid.append(torch.Tensor(vid))
    total_aud = nn.utils.rnn.pad_sequence(total_aud, batch_first=True)
    total_vid = nn.utils.rnn.pad_sequence(total_vid, batch_first=True)
    
    return total_aud, total_vid