import torch
from pathlib import Path
import os


def calculate_kl(featuresdict_1, featuresdict_2, feat_layer_name, same_name=True):
    if not same_name:
        return (
            {
                "kullback_leibler_divergence_sigmoid": float(-1),
                "kullback_leibler_divergence_softmax": float(-1),
            },
            None,
            None,
        )

    # print(
    #     'KL: Assuming that `input2` is "pseudo" target and `input1` is prediction. KL(input2_i||input1_i)'
    # )
    EPS = 1e-6
    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]
    paths_1 = [os.path.basename(x) for x in featuresdict_1["file_path_"]]
    paths_2 = [os.path.basename(x) for x in featuresdict_2["file_path_"]]
    path_to_feats_1 = {p: f for p, f in zip(paths_1, features_1)}
    path_to_feats_2 = {p: f for p, f in zip(paths_2, features_2)}
    sharedkey_to_feats_1 = {p: path_to_feats_1[p] for p in paths_1}
    sharedkey_to_feats_2 = {p: path_to_feats_2[p] for p in paths_2}

    features_1 = []
    features_2 = []

    for sharedkey, feat_2 in sharedkey_to_feats_2.items():
        if sharedkey not in sharedkey_to_feats_1.keys():
            print("%s is not in the generation result" % sharedkey)
            continue
        features_1.extend([sharedkey_to_feats_1[sharedkey]])
        features_2.extend([feat_2])

    features_1 = torch.stack(features_1, dim=0)
    features_2 = torch.stack(features_2, dim=0)

    kl_ref = torch.nn.functional.kl_div(
        (features_1.softmax(dim=1) + EPS).log(),
        features_2.softmax(dim=1),
        reduction="none",
    ) / len(features_1)
    kl_ref = torch.mean(kl_ref, dim=-1)

    # AudioGen use this formulation
    kl_softmax = torch.nn.functional.kl_div(
        (features_1.softmax(dim=1) + EPS).log(),
        features_2.softmax(dim=1),
        reduction="sum",
    ) / len(features_1)

    # For multi-class audio clips, this formulation could be better
    kl_sigmoid = torch.nn.functional.kl_div(
        (features_1.sigmoid() + EPS).log(), features_2.sigmoid(), reduction="sum"
    ) / len(features_1)

    return (float(kl_sigmoid),
        float(kl_softmax),
        kl_ref,
        paths_1,
    )