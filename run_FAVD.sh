#!/bin/bash

# Function to show how to use the script
usage() {
    echo "Usage: bash run_FAVD.sh --ref=<video_paths_ref> --eval=<video_paths_eval>"
    exit 1
}

# Check if no arguments provided
if [ "$#" -eq 0 ]; then
    usage
fi

VIDEO_PATHS_REF=""
VIDEO_PATHS_EVAL=""

# Extract arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ref=*) VIDEO_PATHS_REF="${1#*=}" ;;
        --eval=*) VIDEO_PATHS_EVAL="${1#*=}" ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$VIDEO_PATHS_REF" ] || [ -z "$VIDEO_PATHS_EVAL" ]; then
    usage
fi

# Process REF videos
cd PEAVS/av_features_extraction
python main.py device="cuda:0" video_paths="$VIDEO_PATHS_REF"
cd ../..

mkdir -p FAVD/features
mv PEAVS/av_features_extraction/output_feats/vggish FAVD/features/vggish_ref
mv PEAVS/av_features_extraction/output_feats/i3d_feats FAVD/features/i3d_ref
rm -rf PEAVS/av_features_extraction/output_feats/vggish
rm -rf PEAVS/av_features_extraction/output_feats/i3d_feats

# Process EVAL videos
cd PEAVS/av_features_extraction
python main.py device="cuda:0" video_paths="$VIDEO_PATHS_EVAL"
cd ../..

mv PEAVS/av_features_extraction/output_feats/vggish FAVD/features/vggish_eval
mv PEAVS/av_features_extraction/output_feats/i3d_feats FAVD/features/i3d_eval
rm -rf PEAVS/av_features_extraction/output_feats/vggish
rm -rf PEAVS/av_features_extraction/output_feats/i3d_feats

# Compute FAVD
cd FAVD
python compute_FAVD.py
