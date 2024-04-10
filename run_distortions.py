"""
Media Distortion Generator

Description:
    This script provides functionalities to apply various distortions to audio, video,
    or audio-visual media files. Depending on the specified media type,
    different distortion methods are applied.

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT		

Dependencies:
    environment_distortions.yml

Usage:
    Run the script with appropriate command-line arguments as described in the README file.

Note:
    Before running the script, ensure that all dependencies are installed and accessible.
"""

from mediadistortions.audio_distortions import create_audio_distortions
from mediadistortions.visual_distortions import create_visual_distortions
from mediadistortions.audiovisual_distortions import create_av_distortions
import argparse

def generate_distortions(args):
    if args.media_type == 'audios':
        print('Generating distorted audios')
        create_audio_distortions(args.input_path, args.dest_path, args.audio_distortions)

    elif args.media_type == 'videos':
        print('Generating distorted videos')
        create_visual_distortions(args.input_path, args.dest_path, args.visual_distortions)

    elif args.media_type == 'audiovisual':
        print('Generating distorted audiovisual content')
        create_av_distortions(args.input_path, args.dest_path, args.audiovisual_distortions)

    else:
        print("Please enter a valid media type. Options: audios, videos, audiovisual")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate audio, video, audio-visual distortions.')

    parser.add_argument('--input_path', required=True, help='Path to the folder with media content to add distortions to.')
    parser.add_argument('--dest_path', required=True, help='Path to the folder to save the distorted content.')
    parser.add_argument('--media_type', required=True, choices=['audios', 'videos', 'audiovisual'], help='Specify the type of media to generate distortions to.')

    parser.add_argument('--audio_distortions', type=str, nargs='+',
                        default=['gaussian', 'pops', 'low_pass', 'high_pass', 'quantization', 'griffin_lim', 'griffin_lim_zero',
                                 'mel_filter_wide', 'mel_filter_narrow', 'speed_up', 'slow_down', 'speed_up_PP', 'slow_down_PP',
                                 'reverberation', 'pitch_up', 'pitch_down', 'inter_mutting', 'shuffling'],
                        help='Distortion types for audio content.')

    parser.add_argument('--visual_distortions', type=str, nargs='+',
                        default=['speed_up', 'slow_down', 'black_rectangles', 'gaussian_noise', 'salt_and_pepper_noise',
                                 'frame_swap', 'gaussian_blur', 'random_gaps', 'fragment_shuffle'],
                        help='Distortion types for visual content.')

    parser.add_argument('--audiovisual_distortions', type=str, nargs='+',
                        default=['audio_shift', 'audio_speed_up', 'video_speed_up', 'audio_slow_down', 'video_slow_down',
                                 'inter_muting', 'video_gaps', 'av_flicker', 'fragment_shuffle'],
                        help='Distortion types for audio-visual content.')

    args = parser.parse_args()
    generate_distortions(args)






        
