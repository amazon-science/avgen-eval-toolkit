import os
import torch
import numpy as np
import json

def set_deterministic(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
