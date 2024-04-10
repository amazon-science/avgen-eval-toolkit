'''
Audio-Visual Distortions Generation

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT		

Description:

This code will generated artificial distortions to video inputs
and save the resulting videos in a destination folder

Distortions to be generated:

Temporal Shift Audio
    - Parameters: shift length (sec)
Audio Speed Change up
    - Parameters: speed factor %
Video Speed Change up
    - Parameters: speed factor %
Audio Speed Change down
    - Parameters: speed factor %
Video Speed Change down
    - Parameters: speed factor %
Intermittent Muting
    - Parameters: len . Of mute for every 1 sec
Randomly Sized Gaps
    - Parameters: Video	gap len. and prob. 40%
Fragment Shuffling
    - Parameters: len segments to shuffle
AV Flickering
    - Parameters: gap len. and prob. 40%



'''
import moviepy
import numpy as np
import os
import random
import subprocess
import shutil
from tqdm import tqdm
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.fx.all import volumex
from moviepy.editor import (AudioClip, AudioFileClip, VideoFileClip,
                            concatenate_audioclips, concatenate_videoclips)
from moviepy.video.VideoClip import ColorClip
from moviepy.video.fx.all import speedx
from pydub import AudioSegment


#Temporal misalignment: Shift the audio track either forwards or backwards 
# relative to the video. Most basic form of desynchronization and can be implemented 
# to various degrees, from a few milliseconds to several seconds.
def shift_audio_in_video(video_file, output_file, shift_seconds):
    # Load the video
    video = VideoFileClip(video_file)
    # Extract the audio
    audio = video.audio
    # If the shift is positive, we remove that duration from the beginning and end
    if shift_seconds > 0:
        shifted_audio = audio.subclip(shift_seconds, video.duration-shift_seconds)
    # If the shift is negative, we add silence at the beginning and remove that from the end
    else:
        silence = AudioClip(lambda t: 0, duration=-shift_seconds).set_fps(audio.fps)
        shifted_audio = concatenate_audioclips([silence, audio.subclip(0, video.duration+shift_seconds)])
    # Set the shifted audio on the video clip
    video = video.set_audio(shifted_audio)
    # Write the output
    video.write_videofile(output_file)






import moviepy.editor as mpy
def change_audio_speed(input_file, output_file, speed_factor):
    # Load video
    video = mpy.VideoFileClip(input_file)
    original_audio = video.audio
    # Speed up the audio clip
    new_audio = original_audio.fx(mpy.vfx.speedx, speed_factor)
    if new_audio.duration > video.duration:
        # If so, we shorten the audio to match the video's duration
        new_audio = new_audio.subclip(0, video.duration)
    # Create a new video clip with the original video but with the sped up audio
    new_video = video.set_audio(new_audio)
    # Write the output video file
    new_video.write_videofile(output_file, codec='libx264', audio_codec='aac')




# Video Speed Change: Conversely, you could adjust 
# the speed of the video while leaving the audio track 
# untouched. This would cause the visuals to slowly move out of sync with the audio.
def change_video_speed(input_file, output_file, speed_factor):
    # Load the video
    video = VideoFileClip(input_file)
    # Change the speed of the video
    new_video = video.fx(speedx, speed_factor)
    # Set the audio of the video to the original audio
    if new_video.duration > video.audio.duration:
        new_video = new_video.subclip(0, video.audio.duration)
    new_video = new_video.set_audio(video.audio)
    # Write the output
    new_video.write_videofile(output_file, codec='libx264')




# Intermittent Muting: Introduce intermittent periods of silence in the audio track. 
# This won't necessarily desynchronize the audio and video, but it will 
# disrupt the continuity, which could pose a challenge for some synchronization metrics.
def mute_audio_periods(input_file, output_file, mute_period, mute_duration):
    # Load the video
    video = VideoFileClip(input_file)
    audio = video.audio
    # Mute audio periodically
    audio_segments = []
    t = 0
    while t < audio.duration:
        # Before mute period
        if t + mute_period <= audio.duration:
            audio_segments.append(audio.subclip(t, t + mute_period).fx(volumex, 1.0))
        else:
            audio_segments.append(audio.subclip(t, audio.duration).fx(volumex, 1.0))
            break
        t += mute_period
        # Mute period
        if t + mute_duration <= audio.duration:
            audio_segments.append(audio.subclip(t, t + mute_duration).fx(volumex, 0))
        else:
            audio_segments.append(audio.subclip(t, audio.duration).fx(volumex, 0))
            break
        t += mute_duration
    # Create a new audio clip by concatenating all segments
    new_audio = concatenate_audioclips(audio_segments)
    # Set the new audio on the video clip
    video = video.set_audio(new_audio)
    # Write the output
    video.write_videofile(output_file, codec='libx264')






# Randomly Sized Gaps: Introduce gaps of randomly varying sizes into the 
# video track, which will cause the two to go out of sync.
def insert_random_gaps(input_file, output_file, gap_size, gap_probability):
    # Load the video
    video = VideoFileClip(input_file)
    # Cut the video into segments
    segment_size = gap_size
    num_segments = int(video.duration / segment_size)
    segments = [video.subclip(i * segment_size, (i + 1) * segment_size) for i in range(num_segments)]
    # Create a black frame (gap)
    black_gap = ColorClip((video.size[0], video.size[1]), col=(0,0,0)).set_duration(gap_size)
    # Randomly insert black frames to create gaps
    segments_with_gaps = []
    for segment in segments:
        segments_with_gaps.append(segment)
        if random.random() < gap_probability:
            segments_with_gaps.append(black_gap)
    # Concatenate segments to create the final video
    video_with_gaps = concatenate_videoclips(segments_with_gaps)
    # Set the original audio on the adjusted video
    video_with_gaps = video_with_gaps.set_audio(video.audio)
    # Write the output
    video_with_gaps.write_videofile(output_file, codec='libx264')




# Fragment Shuffling: Break the video and audio into segments and 
# rearrange them. This will maintain the internal synchronization of 
# each segment, but the overall synchronization will be lost.
def shuffle_segments(input_file, output_file, segment_size):
    # Load the video
    video = VideoFileClip(input_file)
    # Cut the video into segments
    num_segments = int(video.duration / segment_size)
    segments = [video.subclip(i * segment_size, (i + 1) * segment_size) for i in range(num_segments)]
    # Shuffle the segments
    random.shuffle(segments)
    # Concatenate segments to create the final video
    final_video = concatenate_videoclips(segments)
    # Write the output
    final_video.write_videofile(output_file, codec='libx264')

import moviepy.editor as mpy
from moviepy.video.VideoClip import ColorClip
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
import random

def insert_av_flicker(input_file, output_file, max_gap_duration, probability):
    # Load video
    video = mpy.VideoFileClip(input_file)
    audio = video.audio
    duration = video.duration

    new_video = []
    new_audio = []

    t = 0
    while t < duration:
        if random.random() < probability:
            # Calculate gap duration
            gap_duration = random.uniform(0, max_gap_duration)

            # Insert gap into video
            gap_clip = ColorClip((video.size[0], video.size[1]), col=[0,0,0], duration=gap_duration)
            new_video.append(gap_clip)

            # Only insert audio gap if duration > 0
            if gap_duration > 0:
                # Calculate number of frames for the gap
                num_frames = max(1, int(gap_duration*audio.fps))

                # Insert gap into audio
                if audio.nchannels == 1:  # Mono audio
                    silent_gap = AudioArrayClip(np.array([0]*num_frames)[np.newaxis, :], audio.fps)
                else:  # Stereo audio
                    silent_gap = AudioArrayClip(np.array([[0, 0]]*num_frames), audio.fps)
                new_audio.append(silent_gap)

            t += gap_duration
        else:
            # Copy original segment
            segment = video.subclip(t, min(t+max_gap_duration, duration))
            new_video.append(segment)

            audio_segment = audio.subclip(t, min(t+max_gap_duration, duration))
            new_audio.append(audio_segment)

            t += segment.duration

    final_video = mpy.concatenate_videoclips(new_video)
    final_audio = mpy.concatenate_audioclips(new_audio)
    fps = video.fps
    final_video.audio = final_audio
    final_video.write_videofile(output_file, codec='libx264', audio_codec='aac', fps=fps)

def format_two_chars(number):
    return f'{number:02}'


def create_av_distortions(input_path, dest_path, distortions)

    root_input = input_path
    dest_root = dest_path


    filenames = os.listdir(root_input)

    if not os.path.exists(dest):
        os.makedirs(dest)

    for fname in tqdm(filenames):
            
            levels = [1, .5, .125, -0.045, -0.1, -.125, -.25, -.5, -1, -2]

            distortion_name = 'audio_shift'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = levels[i]
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # Temporal Shift Audio
                    shift_audio_in_video(input_file, output_file, level)  # Shift audio X seconds forward



            levels = [0.025, .05, .10, .15, .20, .25, .30, .4, .5, .75]

            distortion_name = 'audio_speed_up'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = levels[i] + 1
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # Audio Speed Up 
                    change_audio_speed(input_file, output_file, level)  # Speed up audio by X%


            distortion_name = 'video_speed_up'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = levels[i] + 1
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # Video Speed Up
                    change_video_speed( input_file, output_file, level)  # Speed up video by X%
            
            distortion_name = 'audio_slow_down'


            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = 1 - levels[i]
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # # Audio Speed Down
                    change_audio_speed( input_file, output_file, level)  # Slow down audio by a factor of X

            distortion_name = 'video_slow_down'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = 1 - levels[i]
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # Video Speed Down
                    change_video_speed(input_file, output_file, level)  # Speed down video by X%



            levels = [.01, .025, .05, .1, .2, .3, .5, 1, 2.5, 4]

            distortion_name = 'inter_muting'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = levels[i]
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # Itermittent Muting
                    mute_audio_periods( input_file, output_file, level, .4)  # Mute audio every X seconds, for .25 seconds


            distortion_name = 'video_gaps'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = levels[i]
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # Randomly Sized Gaps Video
                    insert_random_gaps( input_file, output_file, level, 0.4)  # Insert X-second gaps with 10% probability


            distortion_name = 'av_flicker'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = levels[i]
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # AV flickering
                    insert_av_flicker(input_file, output_file, level, 0.4)  # Insert X-second gaps with 20% probability




            levels = [.3, .4, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

            distortion_name = 'fragment_shuffle'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = levels[i]
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    # Fragment Shuffling
                    shuffle_segments(input_file, output_file, level)  # Create segments of X seconds each
