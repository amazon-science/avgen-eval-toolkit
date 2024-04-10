'''
Audio Distortions Generation

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT		

Description:

This code will generated artificial distortions to audio inputs
and save the resulting audios in a destination folder

Distortions to be generated:

Gaussian noise
    - Parameters: sigma
Griffin-Lim/Zero
    - Parameters: iterations
Low-Pass
    - Parameters: iterations
High-Pass
    - Parameters: critical freq.
Mel Filter Narrow
    - Parameters: critical freq.
Mel Filter Wide
    - Parameters: num. bands
Pitch Down/UP
    - Parameters: num. bands
Pops
    - Parameters: % pops
Quantization
    - Parameters: bits
Reverberations
    - Parameters: dampening: delay: 0.25 s echos: 5
Intermittent Mutting
    - Parameters: mute_for mute_probability = 40  
    - mute_every = .4
Shuffling
    - Parameters: segment size
Slow down/PP
    - Parameters: speed factor
Speed up/PP
    - Parameters: speed factor


'''

from pydub import AudioSegment
import numpy as np
import os
from tqdm import tqdm
def add_gaussian_noise(input_file, output_file, sigma):
    # Read the input audio file
    audio = AudioSegment.from_wav(input_file)
    # Convert the audio samples to a NumPy array
    samples = np.array(audio.get_array_of_samples())
    # Generate Gaussian noise with mean=0 and the given sigma
    noise = np.random.normal(0, sigma, samples.shape)
    # Add the noise to the original signal
    distorted_samples = samples + noise
    # Ensure that the values remain within the valid range
    distorted_samples = np.clip(distorted_samples, -32768, 32767).astype(np.int16)
    # Create a new AudioSegment object with the distorted samples
    distorted_audio = AudioSegment(
        distorted_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    # Write the distorted audio to the output file
    distorted_audio.export(output_file, format="wav")
# # Example usage
# input_file = '/Users/sglucas/Documents/Internship/distortions/audios/orig_auds/_4rolMIFkgQ.wav'
# output_file = '/Users/sglucas/Documents/Internship/distortions/audios/output1.wav'
# sigma = .31 # Adjust sigma according to the desired noise level
# sigmas =  [0.0001, 0.00031, 0.001, 0.0031, 0.01, 0.031, 0.1, 0.31, 1, 10, 100]
# add_gaussian_noise(input_file, output_file, sigma)



from pydub import AudioSegment
import numpy as np
def add_pops(input_file, output_file, p):
    # Read the input audio file
    audio = AudioSegment.from_wav(input_file)
    # Convert the audio samples to a NumPy array
    samples = np.array(audio.get_array_of_samples())
    max_value = np.abs(samples).max()
    # Determine the number of samples to change
    num_pops = int(len(samples) * p / 100)
    # Randomly select samples
    pop_indices = np.random.choice(len(samples), num_pops, replace=False)
    # Set half of the selected samples to -max_value, and half to +max_value
    for i in range(num_pops // 2):
        samples[pop_indices[i]] = -max_value
    for i in range(num_pops // 2, num_pops):
        samples[pop_indices[i]] = max_value
    # Create a new AudioSegment object with the modified samples
    distorted_audio = AudioSegment(
        samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    # Write the distorted audio to the output file
    distorted_audio.export(output_file, format="wav")
# # Example usage
# input_file = 'input.wav'
# output_file = 'output_pop.wav'
# p = .001  # Percentage of samples to change; adjust as desired
# percentages = [0.0001, 0.00031, 0.001, 0.0031, 0.01, 0.031, 0.1, 0.31, 1, 10, 100]
# add_pops(input_file, output_file, p)


from pydub import AudioSegment
from scipy.signal import butter, lfilter
import numpy as np
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def apply_filter(input_file, output_file, cutoff, filter_type):
    # Read the input audio file
    audio = AudioSegment.from_wav(input_file)
    # Convert the audio samples to a NumPy array
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    # Apply the filter based on the filter type
    if filter_type == 'low':
        b, a = butter_lowpass(cutoff, sample_rate)
    elif filter_type == 'high':
        b, a = butter_highpass(cutoff, sample_rate)
    filtered_samples = lfilter(b, a, samples)
    # Convert back to the original data type
    filtered_samples = np.clip(filtered_samples, -32768, 32767).astype(np.int16)
    # Create a new AudioSegment object with the filtered samples
    filtered_audio = AudioSegment(
        filtered_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    # Write the filtered audio to the output file
    filtered_audio.export(output_file, format="wav")
# # Example usage
# input_file = 'input.wav'
# output_file_low_pass = 'path_to_output_low_pass.wav'
# output_file_high_pass = 'path_to_output_high_pass.wav'
# cutoff_frequency = 400  # Frequency in Hz
# low_pass_freqs = [4000, 3000, 2000, 1500, 1000, 750, 500, 400, 300]
# high_pass_freqs = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000]
# apply_filter(input_file, output_file_low_pass, cutoff_frequency, 'low')  # Low-pass filter
# apply_filter(input_file, output_file_high_pass, cutoff_frequency, 'high') # High-pass filter


from pydub import AudioSegment
import numpy as np
def apply_quantization(input_file, output_file, bits):
    # Read the input audio file
    audio = AudioSegment.from_wav(input_file)
    # Convert the audio samples to a NumPy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.int32)
    # Scale the samples to be between -1 and 1
    samples = samples / 32768.0
    # Quantize the samples to the desired number of bits
    max_value = 2**(bits - 1) - 1
    samples = np.round(samples * max_value)
    # Scale back to the original range
    samples = samples * (32768.0 / max_value)
    # Convert back to int16
    samples = samples.astype(np.int16)
    # Create a new AudioSegment object with the quantized samples
    quantized_audio = AudioSegment(
        samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    # Write the quantized audio to the output file
    quantized_audio.export(output_file, format="wav")
# # Example usage
# input_file = 'input.wav'
# output_file = 'quantized.wav'
# bit_levels = [9, 8, 7, 6, 5, 4, 3, 2]
# bits = 8  # Number of bits per sample; adjust as desired
# apply_quantization(input_file, output_file, bits)


import librosa
from scipy.io.wavfile import write
import numpy as np
def apply_griffin_lim(input_file, output_file, iterations=60):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    # Compute the magnitude spectrogram
    S_mag = np.abs(librosa.stft(y))
    # Reconstruct using Griffin-Lim
    y_reconstructed = librosa.griffinlim(S_mag, n_iter=iterations)
    # Write the output audio file
    write(output_file, sr, y_reconstructed.astype(np.float32))
# # Example usage
# input_file = 'input.wav'
# output_file = 'griffin_lim.wav'
# iters = [500, 200, 100, 50, 20, 10, 5, 1]
# iterations = 1 # Number of iterations; adjust as desired
# apply_griffin_lim(input_file, output_file, iterations)



from scipy.io.wavfile import write
import librosa
import numpy as np
def griffin_lim_zero(S_mag, iterations=60):
    phase = np.zeros_like(S_mag)
    n_fft = (S_mag.shape[0] - 1) * 2
    for _ in range(iterations):
        S_complex = S_mag * np.exp(1j * phase)
        signal = librosa.istft(S_complex, n_fft=n_fft)
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
    return signal
def apply_griffin_lim_zero(input_file, output_file, iterations=60):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    # Compute the magnitude spectrogram
    S_mag = np.abs(librosa.stft(y))
    # Reconstruct using Griffin-Lim with zero phase initialization
    y_reconstructed = griffin_lim_zero(S_mag, iterations)
    # Write the output audio file
    write(output_file, sr, (y_reconstructed * 32767).astype(np.int16))
# # Example usage
# input_file = 'input.wav'
# output_file = 'griffin_lim_zero.wav'
# iters = [500, 200, 100, 50, 20, 10, 5, 1]
# iterations = 1 # Number of iterations; adjust as desired
# apply_griffin_lim_zero(input_file, output_file, iterations)




from scipy.io.wavfile import write
import librosa
import numpy as np
def apply_mel_filter(input_file, output_file, n_mels=128, fmin=0, fmax=16000):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    # Compute the STFT
    S_complex = librosa.stft(y)
    S_mag, phase = librosa.magphase(S_complex)
    n_fft = (S_mag.shape[0] - 1) * 2
    # Convert to Mel-scale spectrogram
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spectrogram = np.dot(mel_filter, S_mag)
    # Invert the Mel-scale spectrogram
    inv_mel_filter = np.linalg.pinv(mel_filter)
    inverted_spectrogram = np.dot(inv_mel_filter, mel_spectrogram)
    # Reconstruct using the original phase
    reconstructed_complex = inverted_spectrogram * np.exp(1j * phase)
    y_reconstructed = librosa.istft(reconstructed_complex)
    # Write the output audio file
    write(output_file, sr, (y_reconstructed * 32767).astype(np.int16))

# # Example usage for wide Mel filter
# input_file = 'input.wav'
# output_file_wide = 'output_mel_wide.wav'
# num_bands_wide = [264, 128, 64, 32]
# apply_mel_filter(input_file, output_file_wide, n_mels=32, fmin=0, fmax=16000)
# # Example usage for narrow Mel filter
# output_file_narrow = 'output_mel_narrow.wav'
# num_bands_narrow = [264, 128, 64, 32, 16, 8]
# apply_mel_filter(input_file, output_file_narrow, n_mels=32, fmin=60, fmax=6000)




from scipy.io.wavfile import read, write
import numpy as np
def speed_up(input_file, output_file, factor):
    sr, y = read(input_file)
    write(output_file, int(sr * factor), y)
def slow_down(input_file, output_file, factor):
    sr, y = read(input_file)
    write(output_file, int(sr / factor), y)
# # Example usage
# input_file = 'input.wav'
# output_file_speed_up = 'path_to_output_speed_up.wav'
# output_file_slow_down = 'path_to_output_slow_down.wav'

# factors_speed = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1]
# factors_slow = [1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]
# speed_up(input_file, output_file_speed_up, factor=2)  # 2x speed up
# slow_down(input_file, output_file_slow_down, factor=2)  # 2x slow down




import librosa
from scipy.io.wavfile import write
def speed_up_preserving_pitch(input_file, output_file, speed_factor):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    # Speed up the audio (preserving pitch)
    D_fast = librosa.phase_vocoder(D, rate = 1/speed_factor)
    y_fast  = librosa.istft(D_fast, hop_length=512)
    # Write the output audio file
    write(output_file, sr, (y_fast* 32767).astype(np.int16))
# # Example usage
# # input_file = 'path_to_input.wav'
# output_file = 'output_PP_speed_up.wav'
# factors_speed = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1]
# speed_up_preserving_pitch(input_file, output_file, speed_factor=.8)  # 2x speed up



def slow_down_preserving_pitch(input_file, output_file, slow_down_factor):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    # Speed up the audio (preserving pitch)
    D_slow = librosa.phase_vocoder(D, rate = 1/slow_down_factor)
    y_slow  = librosa.istft(D_slow, hop_length=512)
    # Write the output audio file
    write(output_file, sr, (y_slow* 32767).astype(np.int16))
# # Example usage
# # input_file = 'path_to_input.wav'
# output_file = 'output_PP_slow_down.wav'
# factors_slow = [1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]
# slow_down_preserving_pitch(input_file, output_file, slow_down_factor=2)  # 2x slow down\



from scipy.io.wavfile import read, write
import numpy as np
def add_reverberation(input_file, output_file, delay_seconds, dampening, num_echos=5):
    sr, y = read(input_file)
    y = y.astype(np.float32)
    # Calculate the delay in samples
    delay_samples = int(sr * delay_seconds)
    # Create a copy of the original signal
    y_reverb = np.copy(y)
    # Add the echoes
    for echo in range(num_echos):
        # Calculate the current dampening factor
        dampening_factor = dampening ** (echo + 1)
        # Add the delayed and dampened echo
        y_reverb[delay_samples * (echo + 1):] += y[:-delay_samples * (echo + 1)] * dampening_factor
    # Normalize the signal
    y_reverb = (y_reverb / np.max(np.abs(y_reverb)) * 32767).astype(np.int16)
    # Write the result to the output file
    write(output_file, sr, y_reverb)
# # Example usage
# # input_file = 'path_to_input.wav'
# output_file_reverb = 'output_reverb.wav'
# delay_seconds = 0.25  # 0.5 seconds delay
# dampenings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# dampening = 0.8  # 50% dampening
# add_reverberation(input_file, output_file_reverb, delay_seconds, dampening)





import librosa
import soundfile as sf
def pitch_shift_up(input_file, output_file, semitones):
    # Read the audio file
    y, sr = librosa.load(input_file, sr=None)
    # Shift the pitch up by the specified number of semitones
    y_shifted = librosa.effects.pitch_shift(y, sr = sr, n_steps=semitones)
    # Write the result to the output file
    sf.write(output_file, y_shifted, sr)
def pitch_shift_down(input_file, output_file, semitones):
    # Read the audio file
    y, sr = librosa.load(input_file, sr=None)
    # Shift the pitch down by the specified number of semitones
    y_shifted = librosa.effects.pitch_shift(y, sr =sr, n_steps=-semitones)
    # Write the result to the output file
    sf.write(output_file, y_shifted, sr)
# # Example usage
# # input_file = 'path_to_input.wav'
# output_file_up = 'output_pitch_up.wav'
# output_file_down = 'output_pitch_down.wav'
# semitones = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
# semitone = .2
# pitch_shift_up(input_file, output_file_up, semitone)
# pitch_shift_down(input_file, output_file_down, semitone)






import numpy as np
import soundfile as sf
def intermittent_muting(input_file, output_file, mute_probability, mute_every, mute_for):
    # Read the audio file
    y, sr = sf.read(input_file)
    # Calculate the samples corresponding to mute_every and mute_for
    mute_every_samples = int(mute_every * sr)
    mute_for_samples = int(mute_for * sr)
    # Iterate through the audio and set segments to zero with given probability
    for i in range(0, len(y), mute_every_samples):
        if np.random.rand() < mute_probability / 100:
            y[i: i + mute_for_samples] = 0
    # Write the result to the output file
    sf.write(output_file, y, sr)
# # Example usage
# # input_file = 'path_to_input.wav'
# output_file = 'output_inter_muting.wav'
# mute_probability = 40  # 40% chance of muting happening
# mute_every = .4  # Consider muting every 2 seconds
# mute_for = 2.5  # Mute for 0.5 seconds if muting happens
# levels = [.01, .025, .05, .1, .2, .3, .5, 1, 2.5, 4]
# intermittent_muting(input_file, output_file, mute_probability, mute_every, mute_for)



import numpy as np
import soundfile as sf
def fragment_shuffling(input_file, output_file, segment_size):
    # Read the audio file
    y, sr = sf.read(input_file)
    # Calculate the samples corresponding to segment_size
    segment_samples = int(segment_size * sr)
    # Check if the segment size is smaller than the total duration
    if segment_samples > len(y):
        raise ValueError("Segment size is longer than the audio duration.")
    # Break the audio into segments
    segments = [y[i: i + segment_samples] for i in range(0, len(y), segment_samples)]
    # Shuffle the segments
    np.random.shuffle(segments)
    # Concatenate the shuffled segments
    y_shuffled = np.concatenate(segments)
    # If the length doesn't match due to non-even division, truncate the result
    y_shuffled = y_shuffled[:len(y)]
    # Write the result to the output file
    sf.write(output_file, y_shuffled, sr)
# # Example usage
# # input_file = 'path_to_input.wav'
# output_file = 'output_shuffle.wav'
# levels = [.3, .4, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
# segment_size = 1  # Segment size in seconds
# fragment_shuffling(input_file, output_file, segment_size)


def format_two_chars(number):
    return f'{number:02}'

def create_audio_distortions(input_path, dest_path, dist_types):
    dest_root = dest_path

    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    root_input = input_path

    filenames = os.listdir(root_input)

    for fname in tqdm(filenames):

        # GAUSSIAN NOISE
        levels =  [0.0001, 0.00031, 0.001, 0.0031, 0.01, 0.031, 0.1, 0.31, 1, 10, 100]

        distortion_name = 'gaussian'

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

                add_gaussian_noise(input_file, output_file, level)


        # ADDING POPS
        levels = [0.0001, 0.00031, 0.001, 0.0031, 0.01, 0.031, 0.1, 0.31, 1]

        distortion_name = 'pops'

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

                add_pops(input_file, output_file, level)

        # LOW PASS AND HIGH PASS FILTER
        levels = [4000, 3000, 2000, 1500, 1000, 750, 500, 400, 300]

        distortion_name = 'low_pass'

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

                apply_filter(input_file, output_file, level, 'low')  # Low-pass filter



        levels = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000]

        distortion_name = 'high_pass'

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

                apply_filter(input_file, output_file, level, 'high') # High-pass filter





        # QUANTIZATION
        levels = [9, 8, 7, 6, 5, 4, 3, 2]

        distortion_name = 'quantization'

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

                apply_quantization(input_file, output_file, level)


        # GRIFFIN LIM
        levels = [500, 200, 100, 50, 20, 10, 5, 1]

        distortion_name = 'griffin_lim'

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

                apply_griffin_lim(input_file, output_file, level)



        # GRIFFIN LIM ZERO
        levels = [500, 200, 100, 50, 20, 10, 5, 1]

        distortion_name = 'griffin_lim_zero'

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

                apply_griffin_lim_zero(input_file, output_file, level)


        #MEL FILTER WIDE
        levels = [264, 128, 64, 32]

        distortion_name = 'mel_filter_wide'

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

                apply_mel_filter(input_file, output_file, n_mels=level, fmin=0, fmax=16000)

        # MEL FILTER NARROW
        levels = [264, 128, 64, 32, 16, 8]

        distortion_name = 'mel_filter_narrow'

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

                apply_mel_filter(input_file, output_file, n_mels=level, fmin=60, fmax=6000)



        # SPEED UP SLOW DOWN
        levels = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1]

        distortion_name = 'speed_up'

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

                speed_up(input_file, output_file, factor=1/level)  # 2x speed up



        levels = [1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]

        distortion_name = 'slow_down'

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

                slow_down(input_file, output_file, factor=level)  # 2x slow down


        # Pitch preserved speed up
        levels = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1]

        distortion_name = 'speed_up_PP'

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

                speed_up_preserving_pitch(input_file, output_file, speed_factor=level)  # 2x speed up



        # Pitch Perserved slow down
        levels= [1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]

        distortion_name = 'slow_down_PP'

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

                slow_down_preserving_pitch(input_file, output_file, slow_down_factor=level)  # 2x slow down\



        # Reverberation
        delay_seconds = 0.25  # 0.25 seconds delay

        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        distortion_name = 'reverberation'

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

                add_reverberation(input_file, output_file, delay_seconds, level)




        # Pitch Up / Down
        levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]

        distortion_name = 'pitch_up'

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

                pitch_shift_up(input_file, output_file, level)



        distortion_name = 'pitch_down'

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

                pitch_shift_down(input_file, output_file, level)




        #Intermittent Muting
        mute_probability = 40  # 40% chance of muting happening
        mute_every = .4  # Consider muting every 2 seconds

        levels = [.01, .025, .05, .1, .2, .3, .5, 1, 2.5, 4]


        distortion_name = 'inter_mutting'

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

                intermittent_muting(input_file, output_file, mute_probability, mute_every, level)


        # Shuffling
        levels = [.3, .4, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]


        distortion_name = 'shuffling'

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

                fragment_shuffling(input_file, output_file, level)