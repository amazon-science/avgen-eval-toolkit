import numpy as np
from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os
from tqdm import tqdm

def compute_average_vmaf(metrics):
    # Extract the VMAF scores from the input list of metrics
    vmaf_scores = [entry['vmaf'] for entry in metrics]
    
    # Compute the average VMAF score
    average_vmaf = sum(vmaf_scores) / len(vmaf_scores)
    
    return average_vmaf
    

def calculate_vmaf(path_1, path_2):
    print("calculate_VMAF...")

    vmaf_results = []
    for fname in tqdm(os.listdir(path_1)):
        ffqm = FfmpegQualityMetrics(os.path.join(path_1, fname), os.path.join(path_2, fname))
        metrics = ffqm.calculate(["vmaf"])

        vmaf_results.append(compute_average_vmaf(metrics['vmaf']))

    vmaf_results = np.array(vmaf_results)

    vmaf = {}
    vmaf_std = {}

    vmaf['final'] = np.mean(vmaf_results)
    vmaf_std['final'] = np.std(vmaf_results)

    result = {
        "vmaf": vmaf,
        "vmaf_std": vmaf_std,
    }
    return result
