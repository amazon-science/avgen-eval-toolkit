import os
import sys
from . import av_sync_Model
import torch
sys.path.append(os.getcwd())

class ModelWrapper():
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = args.device
        return

    def init_model(self):
        """
        Define model and load pretrained weights
        """

        self.av_sync = av_sync_Model.SyncMetric(self.args)
        self.av_sync.to(self.device)

    
    def feed_forward(self, xa, xv, eval=False, **kwargs):
        """
        Feed forward the model
        """
        def __inference__(self, xa, xv, **kwargs):
            mask_a = kwargs.get("aud_mask", None)
            mask_v = kwargs.get("vid_mask", None)
            
            pred = self.av_sync(xa, xv ,mask_a, mask_v)
            return pred
        
        with torch.no_grad():
            return __inference__(self, xa, xv, **kwargs)
    

    def set_eval(self):
        """
        Set the model to eval mode
        """
        self.av_sync.eval()

    def load_model(self, model_path):
        self.av_sync.load_state_dict(torch.load(model_path+"/final_model.pt"))

