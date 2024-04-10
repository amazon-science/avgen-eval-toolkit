import os
import numpy as np
from scipy import linalg

BASE_PATH = './features/'

def ensure_directory_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def combine_audio_video_data(folder_aud, folder_vid, combined_folder):
    aud_files = sorted(os.listdir(folder_aud))
    vid_files = sorted(os.listdir(folder_vid))

    new_path = os.path.join(BASE_PATH, combined_folder)
    ensure_directory_exists(new_path)


    files_a = [f for f in vid_files if f.endswith(".npy")]

    for file in files_a:
        path_a = os.path.join(folder_vid, file)
        path_b = os.path.join(folder_aud, file.replace('i3d.npy', 'vggish.npy'))

        if os.path.exists(path_b):
            data_a = np.load(path_a)
            data_b = np.load(path_b)

            min_length = min(data_a.shape[0], data_b.shape[0])
            data_a = data_a[:min_length, :]
            data_b = data_b[:min_length, :]

            combined_data = np.hstack((data_a, data_b))

            combined_path = os.path.join(new_path, file.replace('_i3d.npy', '.npy'))
            np.save(combined_path, combined_data)

def calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma

def calculate_favd(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

if __name__ == '__main__':
    # Combine ref audio and video data
    combine_audio_video_data(
        os.path.join(BASE_PATH, 'vggish_ref'),
        os.path.join(BASE_PATH, 'i3d_ref'),
        'combined_av_ref'
    )

    # Combine eval audio and video data
    combine_audio_video_data(
        os.path.join(BASE_PATH, 'vggish_eval'),
        os.path.join(BASE_PATH, 'i3d_eval'),
        'combined_av_eval'
    )

    ref_path = os.path.join(BASE_PATH, 'combined_av_ref/')
    to_eval_path = os.path.join(BASE_PATH, 'combined_av_eval/')

    embd_lst = [np.load(os.path.join(ref_path, vid_id)) for vid_id in sorted(os.listdir(ref_path))]
    gt_embeds = np.concatenate(embd_lst, axis=0)
    mu_gt, sigma_gt = calculate_embd_statistics(gt_embeds)

    embd_lst_eval = [np.load(os.path.join(to_eval_path, vid_id)) for vid_id in sorted(os.listdir(to_eval_path))]
    embeds_eval = np.concatenate(embd_lst_eval, axis=0)
    mu_eval, sigma_eval = calculate_embd_statistics(embeds_eval)

    print(f'FAVD Score: {round(calculate_favd(mu_gt, sigma_gt, mu_eval, sigma_eval), 4)}')
    os.system('rm -rf ./features')
