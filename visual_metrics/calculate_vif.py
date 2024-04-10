import numpy as np
from ffmpeg_quality_metrics import FfmpegQualityMetrics
import os
from tqdm import tqdm


def compute_average_vif(vif_scores):
    # Step 1: Compute the average VIF score for each entry
    per_entry_average = []
    for entry in vif_scores:
        scales_sum = entry['scale_0'] + entry['scale_1'] + entry['scale_2'] + entry['scale_3']
        scales_count = 4  # Since there are four scales in each entry
        per_entry_average.append(scales_sum / scales_count)

    # Step 2: Compute the overall average VIF score
    overall_average_vif = sum(per_entry_average) / len(per_entry_average)

    return overall_average_vif


def calculate_vif(path_1, path_2):
    print("calculate_VIF...")

    vif_results = []
    for fname in tqdm(os.listdir(path_1)):
        ffqm = FfmpegQualityMetrics(os.path.join(path_1, fname), os.path.join(path_2, fname))
        metrics = ffqm.calculate(["vif"])

        vif_results.append(compute_average_vif(metrics['vif']))

    vif_results = np.array(vif_results)

    vif = {}
    vif_std = {}

    vif['final'] = np.mean(vif_results)
    vif_std['final'] = np.std(vif_results)

    result = {
        "vif": vif,
        "vif_std": vif_std,
    }
    return result
