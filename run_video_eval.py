'''
Audio and Visual Evaluation Toolkit

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT		

Description:
Video Evaluation - run_video_eval.py
This toolbox includes the following metrics:
- FVD: Frechet Video Distance
- FID: Frechet Inception distance, realized by inceptionv3
- KID: Kernel Inception Distance
- LPIPS: Learned Perceptual Image Patch Similarity
- MiFID: Memorization-Informed Frechet Inception Distance
- SSIM: Structural Similarity Index Measure
- MS-SSIM: Multi-Scale SSIM
- PSNR: Peak Signal-to-Noise Ratio
- PSNRB: Peak Signal To Noise Ratio With Blocked Effect
- VMAF: Video Multi-Method Assessment Fusion
- VIF: Visual Information Fidelity
- CLIP-Score: Implemented with CLIP VIT model

### Running the metrics
python3 run_video_eval.py --preds_folder /path/to/generated/videos --target_folder /path/to/the/target/videos \
--num_frames {Number of frames in your video or to be used for evaluation} --output path/to/NAME_YOUR_RESULTS_FILE.txt


'''
import os
import cv2
import torch
import torchvision.transforms as transforms
import argparse
from visual_metrics.calculate_fvd import calculate_fvd
from visual_metrics.calculate_fid import calculate_fid
from visual_metrics.calculate_kid import calculate_kid
from visual_metrics.calculate_psnr import calculate_psnr
from visual_metrics.calculate_psnrb import calculate_psnrb
from visual_metrics.calculate_ssim import calculate_ssim
from visual_metrics.calculate_lpips import calculate_lpips
from visual_metrics.calculate_ms_ssim import calculate_ms_ssim
from visual_metrics.calculate_clip import calculate_clip
from visual_metrics.calculate_mifid import calculate_mifid
from visual_metrics.calculate_vmaf import calculate_vmaf
from visual_metrics.calculate_vif import calculate_vif
import json

def load_video_frames(video_path, num_frames, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((frame_size[0], frame_size[1])),
            transforms.CenterCrop((frame_size[0], frame_size[1])),
            transforms.ToTensor()
        ])
        tensor_frame = transform(frame)
        frames.append(tensor_frame)
    cap.release()
    return torch.stack(frames)
    

def load_videos_from_folder(folder_path, num_frames):
    videos_tensor_list_orig = []
    vid_fnames = sorted(os.listdir(folder_path))
    for video_name in vid_fnames:
        video_path = os.path.join(folder_path, video_name)
        video_tensor = load_video_frames(video_path, num_frames)
        videos_tensor_list_orig.append(video_tensor)
    return torch.stack(videos_tensor_list_orig)

def load_videos_with_caps(folder_path, num_frames):
    videos_tensor_list_orig = []
    caps = []
    # open and read the file
    vid_fnames = sorted(os.listdir(folder_path))
    for video_name in vid_fnames:
        fname = video_name[:-4]
        caps.append(fname.replace('_',' '))
        video_path = os.path.join(folder_path, video_name)
        video_tensor = load_video_frames(video_path, num_frames)
        videos_tensor_list_orig.append(video_tensor)
    return torch.stack(videos_tensor_list_orig), caps

def process_fvd(data):
    fvd_final = data["fvd"]["fvd"]["final"]

    result = []
    try:
        fvd_16 = data["fvd"]["fvd"]["[:16]"]
        fvd_24 = data["fvd"]["fvd"]["[:24]"]
        result.append(f'FVD_16: {fvd_16:.3f}\n')
        result.append(f'FVD_24: {fvd_24:.3f}\n')
        result.append(f'FVD: {fvd_final:.3f}\n')
    except:
        result.append(f'FVD: {fvd_final:.3f}\n')
    return ''.join(result)

def process_metric(metric_name, data):
    result = []
    for key, value in data[metric_name][metric_name].items():
        if 'avg' in key:
            avg_key = key
            std_key = key.replace("avg", "std")
            num = avg_key.split('[:')[1].split(']')[0]
            avg_value = value
            std_value = data[metric_name][f"{metric_name}_std"][std_key]
            result.append(f'{metric_name.upper()}_{num}: Average = {avg_value:.3f}, Std = {std_value:.3f}\n')
    final_avg = data[metric_name][metric_name]["final"]
    final_std = data[metric_name][f"{metric_name}_std"]["final"]
    result.append(f'{metric_name.upper()}: Average = {final_avg:.3f}, Std = {final_std:.3f}\n')

    return ''.join(result)

def main(args):
    orig_videos = load_videos_from_folder(args.target_folder, args.num_frames)
    new_videos = load_videos_from_folder(args.preds_folder, args.num_frames)

    results, output_strs = {}, []
    if 'fvd' in args.metrics:
        results['fvd'] = calculate_fvd(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final, args.device)
        print(results)
        output_strs.append(process_fvd(results))
        
    if 'ssim' in args.metrics:
        results['ssim'] = calculate_ssim(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final)
        output_strs.append(process_metric('ssim', results))
    if 'psnr' in args.metrics:
        results['psnr'] = calculate_psnr(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final)
        output_strs.append(process_metric('psnr', results))
    if 'psnrb' in args.metrics:
        results['psnrb'] = calculate_psnrb(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final)
        output_strs.append(process_metric('psnrb', results))
    if 'lpips' in args.metrics:
        results['lpips'] = calculate_lpips(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final, args.net)
        output_strs.append(process_metric('lpips', results))
    if 'ms_ssim' in args.metrics:
        results['ms_ssim'] = calculate_ms_ssim(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final)
        output_strs.append(process_metric('ms_ssim', results))
    if 'fid' in args.metrics:
        results['fid'] = calculate_fid(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final, args.feat_layer)
        output_strs.append(process_metric('fid', results))
    if 'mifid' in args.metrics:
        results['mifid'] = calculate_mifid(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final, args.feat_layer)
        output_strs.append(process_metric('mifid', results))
    if 'kid' in args.metrics:
        results['kid'] = calculate_kid(orig_videos, new_videos, args.calculate_per_frame, args.calculate_final, args.feat_layer, args.subset_size)
        output_strs.append(process_metric('kid', results))
    if 'clip' in args.metrics:
        clip_videos, caps = load_videos_with_caps(args.preds_folder, args.num_frames)
        results['clip'] = calculate_clip(clip_videos, caps, args.calculate_per_frame, args.calculate_final, args.clip_model)
        output_strs.append(process_metric('clip', results))
    if 'vif' in args.metrics:
        results['vif'] = calculate_vif(args.target_folder, args.preds_folder)
        output_strs.append(process_metric('vif', results))
    if 'vmaf' in args.metrics:
        results['vmaf'] = calculate_vmaf(args.target_folder, args.preds_folder)
        output_strs.append(process_metric('vmaf', results))


    with open(args.output, 'w') as file:
        file.writelines(output_strs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate video on visual metrics')

    # Paths
    parser.add_argument('--target_folder', required=True, type=str,
                        help='Path to original videos.')
    parser.add_argument('--preds_folder', required=True, type=str,
                        help='Path to new videos.')

    # Metrics related arguments
    parser.add_argument('--metrics', nargs='+', type=str,
                        default=['fvd', 'ssim', 'psnr', 'psnrb', 'lpips', 'ms_ssim', 'clip', 'fid', 'kid',
                                 'mifid', 'vmaf', 'vif'],
                        help='Metrics to compute. Possible values: fvd, ssim, psnr, psnrb, lpips, ms_ssim, clip, fid, kid, mifid, vmaf, vif')
    parser.add_argument('--net', default='alex', type=str,
                        help="Backbone network type for lpips. Choose between 'alex', 'vgg', or 'squeeze'")
    parser.add_argument('--clip_model', default='openai/clip-vit-base-patch16', type=str,
                        help='Version of the CLIP model to use. Available models: "openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14"')
    parser.add_argument('--feat_layer', default=64, type=int,
                        help='Inceptionv3 feature layer for FID, KID, IS, MIFID. Options: 64, 192, 768, 2048')

    # Frame related arguments
    parser.add_argument('--num_frames', default=16, type=int, help='Number of frames.')
    parser.add_argument('--subset_size', default=16, type=int,
                        help='Frame samples in each video for KID computation')
    parser.add_argument('--calculate_per_frame', default=8, type=int,
                        help='Calculation per frame.')

    # Miscellaneous
    parser.add_argument('--output', default='results.txt', type=str,
                        help='File to save the results.')
    parser.add_argument('--calculate_final', default=True, action='store_true',
                        help='Calculate final metrics.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device for computations. Default is cuda.')

    args = parser.parse_args()
    main(args)