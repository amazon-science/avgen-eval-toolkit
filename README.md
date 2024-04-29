# Audio-Visual Synchrony Evaluation

Paper: [PEAVS: Perceptual Evaluation of Audio-Visual Synchrony Grounded in Viewers' Opinion Scores](https://arxiv.org/abs/2404.07336) 


If you use this work, please cite:

```
@misc{goncalves2024peavs,
      title={PEAVS: Perceptual Evaluation of Audio-Visual Synchrony Grounded in Viewers' Opinion Scores}, 
      author={Lucas Goncalves and Prashant Mathur and Chandrashekhar Lavania and Metehan Cekic and Marcello Federico and Kyu J. Han},
      year={2024},
      eprint={2404.07336},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Environment 
Create the environment from the environment.yml file:
```shell
conda create --name eval_toolkit --file spec-file.txt
source activate eval_toolkit
pip install -r requirements.txt
```

Make sure you have cmd line `FFmpeg 6.0` installed. You can check for installation in the terminal, by typing `ffmpeg -version` and press enter

Activate environment with `source activate eval_toolkit`

## Video Evaluation
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

### Running the metric
```shell
python3 run_video_eval.py --preds_folder /path/to/generated/videos --target_folder /path/to/the/target/videos --num_frames {Number of frames in your video or to be used for evaluation} --output path/to/NAME_YOUR_RESULTS_FILE.txt
```

### NOTE:
- Using `--metrics` you can specify the metrics you want to run `['fvd', 'ssim', 'psnr', 'psnrb', 'lpips', 'ms_ssim', 'clip', 'fid', 'kid', 'mifid', 'vmaf', 'vif']`.
- To run CLIP-Score, name your video as per the caption used to generate it (e.g., `“A horse running on a road”` becomes `A_horse_running_on_a_road.mp4`).
- Ensure the /path/to/generated/videos and /path/to/the/target/videos directories contain an equal number of files with matching names.
- With `--clip_model`, choose the version of the CLIP model to use. Default is `‘openai/clip-vit-base-patch16’`.
- With `--net, specify` the backbone network for LPIPS (options: `‘alex’, ‘vgg’, ‘squeeze’`. Default is ‘alex’).
- With `--feat_layer`, select the inceptionv3 feature layer for FID, KID, and MIFID computation (options: `64, 192, 768, 2048`. Default is 64).
- With `--calculate_per_frame`, set the number of frame steps for each metric computation (e.g., for a 32-frame video and argument of 8, metrics will be computed every 8 frames). Default is 8. All metrics will run over the entire video frames regardless of this argument.
- If you have issues running VMAF please ensure you have cmd line `FFmpeg 6.0` and `ffmpeg-quality-metrics` installed

## Audio Evaluation
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

### Running the metric
```shell
python3 run_audio_eval.py --preds_folder /path/to/generated/audios --target_folder /path/to/the/target_audios --metrics SI_SDR SDR SI_SNR SNR PESQ STOI CLAP FAD ISC FD KL --results NAME_YOUR_RESULTS_FILE.txt
```

### NOTE:
- Using `--metrics`, you can specify the metrics to run `[SI_SDR, SDR, SI_SNR, SNR, PESQ, STOI, CLAP, FAD, ISC, FD, KL]`.
- If no `/path/to/the/target/audios` is provided, only reference-free metrics will be run on your audios.
- For CLAP-Score, name your audio as per the caption used to generate it (e.g., `“A car revving while honking”` becomes `A_car_revving_while_honking.wav`).
- Ensure `/path/to/generated/audios` and `/path/to/the/target/audios` directories contain an equal number of files with matching names. Otherwise, only reference-free metrics will be run.
- Using `--clap_model`, you can can specify the CLAP model id to be used for CLAP score computations. `clap_model = 0 --> 630k non-fusion ckpt; clap_model = 1 --> 630k+audioset non-fusion ckpt; clap_model = 2 --> 630k fusion ckpt; clap_model = 3 --> 630k+audioset fusion ckpt` 
For general audio less than 10-sec: use 0 or 1; For general audio with variable-length: use 2 or 3; more model can be use with a few modifications (please refer to: https:github.com/LAION-AI/CLAP)


## Distortions Generation

This script applies specified distortions to audio, video, or audio-visual media.

## Dependencies

Create the environment from the environment.yml file:
```shell
conda env create -f environment_distortions.yml
```

## Getting Started

2. **Run the script**:
  ```bash
  python3 [YOUR SCRIPT NAME].py --input_path [PATH_TO_MEDIA_FOLDER] --dest_path [PATH_TO_DESTINATION_FOLDER] --media_type [TYPE_OF_MEDIA] [OTHER_OPTIONS]
  ```

### Arguments

- `--input_path`: Path to the folder containing media files you want to apply distortions to.
- `--dest_path`: Path to the destination folder where the distorted media files will be saved.
- `--media_type`: Specify the type of media you want to distort. Options are:
  - `audios`
  - `videos`
  - `audiovisual`

#### Optional Distortion Type Arguments

Specify specific distortion types or use the default ones:
- `--audio_distortions`: List of distortion types for audio content. (e.g., `gaussian`, `pops`, `low_pass`, etc.)
- `--visual_distortions`: List of distortion types for visual content. (e.g., `speed_up`, `black_rectangles`, `gaussian_noise`, etc.)
- `--audiovisual_distortions`: List of distortion types for audio-visual content. (e.g., `audio_shift`, `audio_speed_up`, `video_speed_up`, etc.)

**Note**: Each distortion argument accepts multiple values. For example:
```bash
--audio_distortions gaussian pops low_pass
```

## Example

To apply distortions to audios:
```bash
python [YOUR SCRIPT NAME].py --input_path ./media/audios --dest_path ./distorted_audios --media_type audios --audio_distortions gaussian pops
```

Please replace placeholders like `[YOUR REPO URL]`, `[YOUR REPO DIRECTORY]`, and `[YOUR SCRIPT NAME]` with appropriate values before sharing or using the README.



## PEAVS (Perceptual Evaluation of Audio-Visual Synchrony)
This code will run the trained audio-visual PEAVS metric trained using human perception scores. 

### Running the metric
```shell
bash run_PEAVS.sh videos_folder={/path/to/your/videos/folder}
```

Simply replace the placeholder `{/path/to/your/videos/folder}` with the path to your generated audio-visual content folder.
Results are saved in `./PEAVS/results/results.csv`


## FAVD (Frechet Audio-Visual Distance)

### Running the metric
```shell
bash run_FAVD.sh --ref=<video_paths_ref> --eval=<video_paths_eval>
```

Simply replace the placeholder `<video_paths_ref>` with the path to the reference audio-visual content you will be using
and replace the placeholder `<video_paths_eval>` with the path to the audio-viusal content you want to evaluate with FAVD.

## AVS Benchmark Dataset

The AVS Benchmark Dataset is designed for the development and evaluation of metrics that assess audio-visual (AV) synchronization in multimedia content. This benchmark focuses on human perception of AV sync errors under various distortion scenarios.

### Data Composition

The dataset comprises 200 videos from the AudioSet corpus, carefully selected to exclude any content with faces and to represent a range of synchrony-sensitive activities. Each video is subjected to nine different types of synchrony-related distortions at ten levels of intensity, resulting in 18,200 unique distorted videos.

### Distortion Types

Distortions applied aim to mimic real-world sync anomalies caused by factors such as network issues or encoding errors. These include:
- Temporal Misalignment
- Audio Speed Change
- Video Speed Change
- Fragment Shuffling
- Intermittent Muting
- Randomly Sized Gaps
- AV Flickering

Each type of distortion is crafted to either create de-synchronization (temporal noise) or present challenges in perceptual sync (static noise) without actual desynchronization.

### Annotation Process

The dataset avoids evaluating the quality of audio or video in isolation, focusing solely on AV sync issues. A pairwise comparison approach is employed, where annotators rate videos side-by-side based on synchronization quality. Over 60,000 pairwise ratings were collected, ensuring each video was rated at least three times for robustness.
