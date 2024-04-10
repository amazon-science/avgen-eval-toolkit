#!/bin/bash

# Extract videos_folder argument
for arg in "$@"
do
    case $arg in
        videos_folder=*)
        VIDEO_PATHS="${arg#*=}"
        shift # Remove video_folder= from processing
        ;;
        *)
        shift # Remove generic argument from processing
        ;;
    esac
done

cd PEAVS
cd av_features_extraction
python main.py device="cuda:0" video_paths=$VIDEO_PATHS
cd ..
python metric.py