import os
import csv
import glob
import json
import numpy as np
import pandas as pd

"""
"""


def load_env(env_path):
    with open(env_path, 'r') as f:
        env_dict = json.load(f)
    return env_dict

class DataManager:
    def __init__(self):
        self.msp_label_dict = None

    def get_aud_path(self, aud_loc=None, fnames =[], *args, **kwargs):

        paths_aud = []
        for aud_name in fnames:
            paths_aud.append(aud_loc + '/' + aud_name)
        paths_aud.sort()
        return paths_aud

    def get_vid_path(self, vid_loc=None, fnames = [], *args, **kwargs):

        paths_vid, names = [], []

        for vid_name in fnames:
            paths_vid.append(vid_loc + '/' + vid_name)
            names.append(vid_name.replace('_i3d.npy',''))
        paths_vid.sort()
        names.sort()

        return paths_vid, names