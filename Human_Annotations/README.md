# AVS Benchmark Dataset

The AVS Benchmark Dataset is designed for the development and evaluation of metrics that assess audio-visual (AV) synchronization in multimedia content. This benchmark focuses on human perception of AV sync errors under various distortion scenarios.

This section contains instructions to download the dataset, replicate the distortions, and obtain the annotations.

### Data Composition

The dataset comprises 200 videos from the AudioSet corpus, carefully selected to exclude any content with faces and to represent a range of synchrony-sensitive activities. Each video is subjected to nine different types of synchrony-related distortions at ten levels of intensity, resulting in 18,200 unique distorted videos.

To obtain the raw video files from AudioSet, please refer to this [link](https://research.google.com/audioset/download.html). After obtaining the AudioSet files, the filenames to be used to replicate our dataset are provided in the file [Human_Annotations/audioset_filenames.csv](https://github.com/amazon-science/avgen-eval-toolkit/blob/main/Human_Annotations/audioset_filenames.csv).

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

#### Generating the AVS Benchmark Dataset Files

To generate all the distorted files provided in the AVS Benchmark Dataset, please refer to the [Distortions Generation](https://github.com/amazon-science/avgen-eval-toolkit/tree/main?tab=readme-ov-file#distortions-generation) section of this document.

To apply distortions to the 200 files provided in our dataset, please run the following:

```bash
python run_distortions.py --input_path ./path/to/the/original_videos --dest_path ./distorted_videos --media_type audiovisual
```

This code will automatically output the files for our dataset and save them in the `./distorted_videos` location.

### Annotation Process

The dataset avoids evaluating the quality of audio or video in isolation, focusing solely on AV sync issues. A pairwise comparison approach is employed, where annotators rate videos side-by-side based on synchronization quality. Over 60,000 pairwise ratings were collected, ensuring each video was rated at least three times for robustness.

### Obtaining AVS Benchmark Annotations

The raw annotations can be directly found under [Human_Annotations](https://github.com/amazon-science/avgen-eval-toolkit/blob/main/Human_Annotations).
Additionally, training, testing, and development splits can be found under [/Human_Annotations/training_files](https://github.com/amazon-science/avgen-eval-toolkit/tree/main/Human_Annotations/training_files).
