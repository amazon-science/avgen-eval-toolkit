# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse

# 3rd-Party Modules
import numpy as np
from tqdm import tqdm
import pandas as pd

# PyTorch Modules
import torch
from torch.utils.data import DataLoader


# Self-Written Modules
sys.path.append(os.getcwd())
import utils
import net

def main(args):
    utils.set_deterministic(args.seed)
    
    # Initialize dataset
    DataManager = utils.DataManager()

    audio_path = './av_features_extraction/output_feats/vggish'
    video_path = './av_features_extraction/output_feats/i3d_feats'

    fnames_aud = sorted(os.listdir(audio_path))
    fnames_vid = sorted(os.listdir(video_path))

    assert len(fnames_aud) == len(fnames_vid), f"fnames_aud: {len(fnames_aud)} is not equal to fnames_vid: {len(fnames_vid)}"

    test_aud_path = DataManager.get_aud_path(aud_loc=audio_path, fnames=fnames_aud)
    test_vid_path, filenames = DataManager.get_vid_path(vid_loc=video_path, fnames=fnames_vid)

    test_auds = utils.AudExtractor(test_aud_path).extract()
    test_vids = utils.VidExtractor(test_vid_path).extract()
    ###################################################################################################

    model_path = "./metric_weights"   

    test_set = utils.AudVidSet(test_auds, test_vids,
        print_dur=True)

    batch_size=args.batch_size
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=utils.collate_fn_padd, shuffle=False)
    
    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.load_model(model_path)
    modelWrapper.set_eval()
 

    with torch.no_grad():
        total_pred = [] 
        for xy_pair in tqdm(test_loader):
            xa = xy_pair[0]
            xv = xy_pair[1]
            mask_a = xy_pair[2]
            mask_v = xy_pair[3]
        
            xa=xa.cuda(non_blocking=True).float()
            xv=xv.cuda(non_blocking=True).float()
            mask_a=mask_a.cuda(non_blocking=True).float()
            mask_v=mask_v.cuda(non_blocking=True).float()

            pred = modelWrapper.feed_forward(xa, xv, aud_mask=mask_a, vid_mask=mask_v)

            total_pred.append(torch.clamp(pred*4+1, min=1, max=5))

        total_pred = torch.cat(total_pred, 0)
    
    total_pred_np = total_pred.detach().cpu().numpy().ravel()

    # Create a DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'score': total_pred_np
    })

        
    save_path = './results/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save to CSV
    df.to_csv(save_path + 'results.csv', index=False)
    
    print("#--------------------------------------------------------------------#")
    print(f"Set score -> Mean: {np.mean(df['score']):.2f}, STD: {np.std(df['score']):.2f}")
    print("#--------------------------------------------------------------------#")

    os.system('rm -rf ./av_features_extraction/output_feats/')

    


if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)


    # Model Arguments
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int)
    parser.add_argument(
        '--output_dim',
        default=1,
        type=int)
    # Transformers Arguments
    parser.add_argument(
        '--attn_dropout', type=float, default=0.1,
        help='attention dropout')
    parser.add_argument(
        '--relu_dropout', type=float, default=0.1,
        help='relu dropout')
    parser.add_argument(
        '--embed_dropout', type=float, default=0.25,
        help='embedding dropout')
    parser.add_argument(
        '--res_dropout', type=float, default=0.1,
        help='residual block dropout')
    parser.add_argument(
        '--out_dropout', type=float, default=0.2,
        help='output layer dropout (default: 0.2')
    parser.add_argument(
        '--layers', type=int, default = 3,
        help='number of layers in the network (default: 5)')
    parser.add_argument(
        '--num_heads', type=int, default = 8,
        help='number of heads for multi-head attention layers(default: 10)')
    parser.add_argument(
        '--attn_mask', action='store_false',
        help='use attention mask for transformer (default: true)')

    args = parser.parse_args()

    # Call main function
    main(args)


