'''
Visual Distortions Generation

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT	

Description:

This code will generated artificial distortions to video inputs
and save the resulting videos in a destination folder

Distortions to be generated:

Black rectangle	size
    - Parameters: relative to image %
Gaussian noise
    - Parameters: percentage of noise in convex combination	
Salt & Pepper
    - Parameters: probability of applying salt or pepper
Local swap
    - Parameters: number of swaps	1,5,10,20,30,40	6
Gaussian blur
    - Parameters: sigma of Gaussian kerne
Video Speed Change up
    - Parameters: speed factor %
Video Speed Change down
    - Parameters: speed factor %
Randomly Sized Gaps Video
    - Parameters: gap len. and prob. 40%
Fragment Shuffling
    - Parameters: len segments to shuffle	


'''
import cv2
import numpy as np
import random
import moviepy
import os
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
from tqdm import tqdm

def black_rectangles(input_video_path, output_video_path, distortion_level):
    # Check if distortion level is within a valid range (0-100)
    if not (0 <= distortion_level <= 100):
        raise ValueError("Distortion level must be between 0 and 100 inclusive.")
    # Read the video
    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Calculate the size of the rectangle based on distortion level
        rect_width = int(width * (distortion_level / 100))
        rect_height = int(height * (distortion_level / 100))
        # Calculate the random starting coordinates for the rectangle
        start_x = random.randint(0, width - rect_width)
        start_y = random.randint(0, height - rect_height)
        # Draw the rectangle on the frame
        cv2.rectangle(frame, (start_x, start_y), (start_x + rect_width, start_y + rect_height), (0, 0, 0), -1)
        # Write the frame with the rectangle to the output video
        out.write(frame)
    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
# # Example of how to use the function
# black_rectangles(input_loc, output_loc, 15)  # Adds a rectangle which takes up 10% of the frame's size



def gaussian_noise(input_video_path, output_video_path, distortion_level):
    # Check distortion level range
    distortion_level = distortion_level/100
    if not (0 <= distortion_level <= 1):
        raise ValueError("Distortion level must be between 0 and 1 inclusive.")
    # Read the video
    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    alpha = 1 - distortion_level
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Generate Gaussian noise
        mean = 0
        var = 25
        sigma = var**0.5
        noise = np.random.normal(mean, sigma, (height, width, 3))
        noise = noise.reshape(height, width, 3)
        # Create noisy frame using convex combination
        noisy_frame = cv2.addWeighted(frame, alpha, noise.astype(np.uint8), (1 - alpha), 0)
        # Write the noisy frame to the output video
        out.write(noisy_frame)
    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
# # Example of how to use the function
# gaussian_noise(input_loc, output_loc, 5)  # 20% of the frame will be noise



def salt_and_pepper_noise(input_video_path, output_video_path, distortion_level):
    # Check distortion level range
    distortion_level = distortion_level/100
    if not (0 <= distortion_level <= 1):
        raise ValueError("Distortion level must be between 0 and 1 inclusive.")
    # Read the video
    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Introduce salt & pepper noise
        noise = np.random.choice([0, 1, 2], size=(height, width, 1), p=[distortion_level/2, distortion_level/2, 1-distortion_level])
        noise = np.repeat(noise, 3, axis=2)
        frame[noise == 0] = 0    # pepper
        frame[noise == 1] = 255  # salt
        # Write the noisy frame to the output video
        out.write(frame)
    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
# # Example of how to use the function
# salt_and_pepper_noise(input_loc, output_loc, 1)  # 5% probability for salt and 5% for pepper



def frame_swap(input_video_path, output_video_path, num_swaps):
    # Read the video
    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Read all frames into a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    total_frames = len(frames)
    # Validate number of swaps
    if num_swaps >= total_frames:
        raise ValueError("Number of swaps should be less than the total number of frames.")
    # Perform swaps
    for _ in range(num_swaps):
        idx = random.randint(0, total_frames - 2)  # Exclude the last frame
        frames[idx], frames[idx + 1] = frames[idx + 1], frames[idx]
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    # Write modified frames to the output video
    for frame in frames:
        out.write(frame)
    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
# # Example of how to use the function
# frame_swap(input_loc, output_loc, 10)  # Swap 10 pairs of frames



def gaussian_blur(input_video_path, output_video_path, distortion_level):
    # Check if distortion level is within a valid range (positive value)
    if distortion_level <= 0:
        raise ValueError("Distortion level (sigma) must be a positive value.")
    # Read the video
    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Applying Gaussian Blur. Kernel size is calculated based on sigma (distortion_level)
        # The kernel size should be odd, hence we multiply distortion_level by 2 and add 1
        ksize = (2*int(distortion_level) + 1, 2*int(distortion_level) + 1)
        blurred_frame = cv2.GaussianBlur(frame, ksize, distortion_level)
        # Write the blurred frame to the output video
        out.write(blurred_frame)
    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
# Example of how to use the function
# gaussian_blur(input_loc, output_loc, 1)  # Adjust the sigma (distortion_level) for desired blur



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


def format_two_chars(number):
    return f'{number:02}'

def create_audio_distortions(input_path, dest_path, dist_types):

    dest_root = dest_path

    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    root_input = input_path

    filenames = os.listdir(root_input)

    for fname in tqdm(filenames):

        try: 

            # SPEED UP
            levels = [0.025, .05, .10, .15, .20, .25, .30, .4, .5, .75]

            distortion_name = 'speed_up'

            if distortion_name in dist_types:

                dest_distortion = os.path.join(dest_root, distortion_name)

                if not os.path.exists(dest_distortion):
                    os.makedirs(dest_distortion)

                val = 0
                for i in range(len(levels)):
                    level = 1 + levels[i]
                    val += 1

                    level_name = 'level_' + format_two_chars(val)

                    dest_level = os.path.join(dest_distortion, level_name)

                    if not os.path.exists(dest_level):
                        os.makedirs(dest_level)
                    
                    output_file = os.path.join(dest_level, fname)
                    input_file =  os.path.join(root_input, fname)

                    change_video_speed(input_file, output_file, level)


            # SLOW DOWN
            levels = [0.025, .05, .10, .15, .20, .25, .30, .4, .5, .75]

            distortion_name = 'slow_down'

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

                    change_video_speed(input_file, output_file, level)



            # BLACK RECTANGLES
            levels =  [1,5,10,20,30,40,50,60,70,80]

            distortion_name = 'black_rectangles'

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

                    black_rectangles(input_file, output_file, level)


            # GAUSSIAN NOISE

            distortion_name = 'gaussian_noise'

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

                    gaussian_noise(input_file, output_file, level)

            #SALT AND PEPPER
            distortion_name = 'salt_and_pepper_noise'

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

                    salt_and_pepper_noise(input_file, output_file, level) 



            #FRAME SWAPPING
            levels =  [2,5,10,20,30,40]
            distortion_name = 'frame_swap'
            
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

                    frame_swap(input_file, output_file, level)


            # GAUSSIAN BLUR
            levels = [1,2,3,4,5]

            distortion_name = 'gaussian_blur'

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

                    gaussian_blur(input_file, output_file, level)



            # RANDOMLY SIZED GAPS
            levels = [.01, .025, .05, .1, .2, .3, .5, 1, 2.5, 4]

            distortion_name = 'random_gaps'

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

                    insert_random_gaps( input_file, output_file, level, 0.4)


            #FRAGMENT SHUFFLING
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

                    shuffle_segments( input_file, output_file, level)

        except:
            print(fname, 'FAILED')

