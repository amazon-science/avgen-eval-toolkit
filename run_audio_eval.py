'''
Audio and Visual Evaluation Toolkit

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT		

Description:
Audio Evaluation - run_audio_eval.py
This toolbox includes the following metrics:
- FAD: Frechet audio distance
- ISc: Inception score
- FD: Frechet distance, realized by PANNs, a state-of-the-art audio classification model
- KL: KL divergence (softmax over logits)
- KL_Sigmoid: KL divergence (sigmoid over logits)
- SI_SDR: Scale-Invariant Signal-to-Distortion Ratio
- SDR: Signal-to-Distortion Ratio
- SI_SNR: Scale-Invariant Signal-to-Noise Ratio
- SNR: Signal-to-Noise Ratio
- PESQ: Perceptual Evaluation of Speech Quality
- STOI: Short-Time Objective Intelligibility
- CLAP-Score: Implemented with LAION-AI/CLAP

### Running the metris
python run_audio_eval.py --preds_folder /path/to/generated/audios --target_folder /path/to/the/target_audios \
--metrics SI_SDR SDR SI_SNR SNR PESQ STOI CLAP FAD ISC FD KL --results NAME_YOUR_RESULTS_FILE.txt


Third-Party Snippets/Credits:

[1] - Taken from [https://github.com/haoheliu/audioldm_eval] - [MIT License]
    - Adapted code for FAD, ISC, FID, and KL computation

[2] - Taken from [https://github.com/LAION-AI/CLAP] - [CC0-1.0 license]
    - Snipped utilized for audio embeddings and text embeddings retrieval

'''
import argparse
import os
import numpy as np
import datetime
import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.audio import (ScaleInvariantSignalDistortionRatio, ScaleInvariantSignalNoiseRatio,
                                SignalDistortionRatio, SignalNoiseRatio, PerceptualEvaluationSpeechQuality,
                                ShortTimeObjectiveIntelligibility)
from utils.load_mel import WaveDataset
import laion_clap
from audio_metrics.clap_score import calculate_clap
from audio_metrics.fad import FrechetAudioDistance
from audio_metrics.fid import calculate_fid
from audio_metrics.isc import calculate_isc
from audio_metrics.kl import calculate_kl
from feature_extractors.panns import Cnn14


def check_folders(preds_folder, target_folder):
    preds_files = [f for f in os.listdir(preds_folder) if f.endswith('.wav')]
    target_files = [f for f in os.listdir(target_folder) if f.endswith('.wav')]

    if len(preds_files) != len(target_files):
        print('Mismatch in number of files between preds and target folders.')
        return False

    if set(preds_files) != set(target_files):
        print('Mismatch in filenames between preds and target folders.')
        return False
    
    return True

def get_current_time():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H:%M:%S')

def get_featuresdict( dataloader, device, mel_model):
    out = None
    out_meta = None

    for waveform, filename in tqdm(dataloader):
        
        metadict = {
            'file_path_': filename,
        }
        waveform = waveform.squeeze(1)

        waveform = waveform.float().to(device)

        with torch.no_grad():
            featuresdict = mel_model(waveform) # 'logits': [1, 527]

        featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

        if out is None:
            out = featuresdict
        else:
            out = {k: out[k] + featuresdict[k] for k in out.keys()}

        if out_meta is None:
            out_meta = metadict
        else:
            out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    return {**out, **out_meta}

def evaluate_audio_metrics(preds_folder, target_folder, metrics, results_file, clap_model):
    scores = {metric: [] for metric in metrics}
    
    if target_folder == None or not check_folders(preds_folder, target_folder):
        text = 'Running only reference-free metrics'
        same_name = False

    elif check_folders(preds_folder, target_folder):
        text = 'Running all metrics specified'
        same_name = True

        # Initialize the specified metrics
        si_sdr = ScaleInvariantSignalDistortionRatio() if 'SI_SDR' in metrics else None
        sdr_calculator = SignalDistortionRatio() if 'SDR' in metrics else None
        si_snr = ScaleInvariantSignalNoiseRatio() if 'SI_SNR' in metrics else None
        snr_calculator = SignalNoiseRatio() if 'SNR' in metrics else None
        pesq_metric = PerceptualEvaluationSpeechQuality(16000, 'wb') if 'PESQ' in metrics else None
        fs = 16000
        stoi_metric = ShortTimeObjectiveIntelligibility(fs, extended=False) if 'STOI' in metrics else None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'FAD' in metrics or 'KL' in metrics or 'ISC' in metrics or 'FD' in metrics:

        backbone = 'cnn14'
        sampling_rate = 16000
        frechet = FrechetAudioDistance()
        
        frechet.model = frechet.model.to(device)

        if sampling_rate == 16000:
            mel_model = Cnn14(
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
        else:
            raise ValueError(
                'We only support the evaluation on 16kHz sampling rate.'
            )

        mel_model.eval()
        mel_model.to(device)
        fbin_mean, fbin_std = None, None

    
        torch.manual_seed(0)

        num_workers = 6

        outputloader = DataLoader(
            WaveDataset(
                preds_folder,
                sampling_rate, 
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )

        resultloader = DataLoader(
            WaveDataset(
                target_folder,
                sampling_rate, 
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )

        out = {}

        # FAD
        if 'FAD' in metrics:
            fad_score = frechet.score(preds_folder, target_folder, limit_num=None)
            out['frechet_audio_distance'] = fad_score
        
            print('Extracting features from %s.' % target_folder)
            featuresdict_2 = get_featuresdict(resultloader, device, mel_model)
            
            print('Extracting features from %s.' % preds_folder)
            featuresdict_1 = get_featuresdict(outputloader, device, mel_model)

        if check_folders(preds_folder, target_folder) and 'KL' in metrics:
            kl_sigmoid, kl_softmax, kl_ref, paths_1 = calculate_kl(
                featuresdict_1, featuresdict_2, 'logits', same_name
            )
            out['kullback_leibler_divergence_sigmoid'] = float(kl_sigmoid)
            out['kullback_leibler_divergence_softmax'] =  float(kl_softmax)


        if 'ISC' in metrics:
            print('Extracting features from %s.' % preds_folder)
            featuresdict_1 = get_featuresdict(outputloader, device, mel_model)

            mean_isc, std_isc = calculate_isc(
                featuresdict_1,
                feat_layer_name='logits',
                splits=10,
                samples_shuffle=True,
                rng_seed=2020,
            )
            out['inception_score_mean'] =  mean_isc
            out['inception_score_std'] = std_isc


        if 'FD' in metrics:
            print('Extracting features from %s.' % target_folder)
            featuresdict_2 = get_featuresdict(resultloader, device, mel_model)
            
            print('Extracting features from %s.' % preds_folder)
            featuresdict_1 = get_featuresdict(outputloader, device, mel_model)

            if('2048' in featuresdict_1.keys() and '2048' in featuresdict_2.keys()):
                metric_fid = calculate_fid(
                    featuresdict_1, featuresdict_2, feat_layer_name='2048'
                )
                out['frechet_distance'] = round(metric_fid, 3)


    # Loading Clap Model
    if 'CLAP' in metrics:
        if clap_model == 0 or clap_model == 1:
            model_clap = laion_clap.CLAP_Module(enable_fusion=False) 
        elif clap_model == 2 or clap_model == 3:
            model_clap = laion_clap.CLAP_Module(enable_fusion=True) 

        model_clap.load_ckpt(model_id=clap_model) # Download the default pretrained checkpoint.
        # Resampling rate
        new_freq = 48000
    else:
        model_clap = None


    # Get the list of filenames and set up the progress bar
    filenames = [f for f in os.listdir(preds_folder) if f.endswith('.wav')]
    progress_bar = tqdm(filenames, desc='Processing')

    print(text)
    for filename in progress_bar:
        if filename.endswith('.wav'):
            try:
                preds_audio, _ = torchaudio.load(os.path.join(preds_folder, filename), num_frames=160000)
                target_audio, _ = torchaudio.load(os.path.join(target_folder, filename), num_frames=160000)
                min_len = min(preds_audio.size(1), target_audio.size(1))
                preds_audio, target_audio = preds_audio[:, :min_len], target_audio[:, :min_len]
                if np.shape(target_audio)[0] == 2:
                    target_audio = target_audio.mean(dim=0)
                if np.shape(preds_audio)[0] == 2:
                    preds_audio = preds_audio.mean(dim=0)

                # Compute and store the scores for the specified metrics
                if 'CLAP' in metrics: scores['CLAP'].append(calculate_clap(model_clap, preds_audio, filename, new_freq))
                if si_snr: scores['SI_SNR'].append(si_snr(preds_audio.squeeze(), target_audio.squeeze()).item())
                if snr_calculator: scores['SNR'].append(snr_calculator(preds_audio.squeeze(), target_audio.squeeze()).item())
                if sdr_calculator: scores['SDR'].append(sdr_calculator(preds_audio.squeeze(), target_audio.squeeze()).item())
                if si_sdr: scores['SI_SDR'].append(si_sdr(preds_audio.squeeze(), target_audio.squeeze()).item())
                if pesq_metric: scores['PESQ'].append(pesq_metric(preds_audio.squeeze(), target_audio.squeeze()).item())
                if stoi_metric: scores['STOI'].append(stoi_metric(preds_audio.squeeze(), target_audio.squeeze()).item())

            except:
                print('Error in:', filename)


    # Print and save the average and standard deviation for each metric
    with open(results_file, 'w') as file:
        for metric, values in scores.items():
            if str(metric.upper()) not in ['FAD', 'ISC', 'FD', 'KL']:
                avg = np.mean(values)
                std = np.std(values)
                print(f'{metric.upper()}: Average = {avg}, Std = {std}')
                file.write(f'{metric.upper()}: Average = {avg}, Std = {std}\n')
        if 'FAD' in metrics:
            print(f'FAD: {out['frechet_audio_distance']:.5f}')
            file.write(f'FAD: {out['frechet_audio_distance']:.5f}\n')
        if 'ISC' in metrics:
            print(f'ISc: Average = {out['inception_score_mean']:8.5f}, Std = {out['inception_score_std']:5f})')
            file.write(f'ISc: Average = {out['inception_score_mean']:8.5f}, Std = {out['inception_score_std']:5f}\n')
        if 'FD' in metrics:
            print(f'FD: {out['frechet_distance']:8.5f}')
            file.write(f'FAD: {out['frechet_distance']:8.5f}\n')
        if check_folders(preds_folder, target_folder) and 'KL' in metrics:
            print(f'KL_Sigmoid: {out['kullback_leibler_divergence_sigmoid']:8.5f}')
            print(f'KL_Softmax: {out['kullback_leibler_divergence_softmax']:8.5f}')
            file.write(f'KL_Sigmoid: {out['kullback_leibler_divergence_sigmoid']:8.5f}\n')
            file.write(f'KL: {out['kullback_leibler_divergence_softmax']:8.5f}\n')


# Defining clap model descriptions
CLAP_MODEL_DESCRIPTIONS = {
    0: '630k non-fusion ckpt',
    1: '630k+audioset non-fusion ckpt',
    2: '630k fusion ckpt',
    3: '630k+audioset fusion ckpt'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate audio on acoustic metrics.')

    # Audio paths
    parser.add_argument('--preds_folder', required=True,
                        help='Path to the folder with predicted audio files.')
                        
    parser.add_argument('--target_folder', required=False, default=None,
                        help='Path to the folder with target audio files.')

    # Metrics related
    parser.add_argument('--metrics', nargs='+',
                        choices=['SI_SDR', 'SDR', 'SI_SNR', 'SNR', 'PESQ', 'STOI', 'CLAP', 'FAD', 'ISC', 'FD', 'KL'],
                        help='List of metrics to calculate.')

    # CLAP model selection
    parser.add_argument('--clap_model', type=int, default=1,
                        help=f'CLAP model id for score computations. Options: '
                             f'{', '.join([f'{key} --> {value}' for key, value in CLAP_MODEL_DESCRIPTIONS.items()])}')

    # Results path
    parser.add_argument('--results_file', required=True,
                        help='Path to the text file to save the results.')

    args = parser.parse_args()
    evaluate_audio_metrics(args.preds_folder, args.target_folder, args.metrics, args.results_file, args.clap_model)