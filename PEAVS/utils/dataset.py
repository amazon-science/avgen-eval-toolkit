import torch
import torch.nn as nn
import torch.utils as torch_utils
import numpy as np

class AudVidSet(torch_utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super(AudVidSet, self).__init__()
        self.aud_list = kwargs.get("aud_list", args[0]) # (N, D, T)
        self.vid_list = kwargs.get("vid_list", args[1])

    def __len__(self):
        return len(self.aud_list)

    def __getitem__(self, idx):
        cur_aud = self.aud_list[idx]
        cur_vid = self.vid_list[idx]

        cur_dur_a = len(cur_aud)
        cur_dur_v = len(cur_aud)

        result = (cur_aud, cur_vid, cur_dur_a, cur_dur_v)
        return result

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    total_aud = []
    total_vid = []
    total_dur_a = []
    total_dur_v = []
    for aud, vid, dur_a, dur_v in batch:
        # print('collate',np.shape(aud), np.shape(vid), lab, dur)
        total_aud.append(torch.Tensor(aud))
        total_vid.append(torch.Tensor(vid))
        total_dur_a.append(dur_a)
        total_dur_v.append(dur_v)
    total_aud = nn.utils.rnn.pad_sequence(total_aud, batch_first=True)
    total_vid = nn.utils.rnn.pad_sequence(total_vid, batch_first=True)

    
    max_dur = np.max(total_dur_a)
    mask_a = torch.zeros(total_aud.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur_a):
        mask_a[data_idx,:dur] = 1

    max_dur = np.max(total_dur_v)
    mask_v = torch.zeros(total_vid.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur_v):
        mask_v[data_idx,:dur] = 1
    
    return total_aud, total_vid, mask_a, mask_v