import os
import noisereduce
import scipy.io.wavfile as wav
import numpy as np
from moviepy import VideoFileClip

videoFile = 'Fruit Animation.mp4'
outputDirectory = 'audio'
noisyAudioPath = os.path.join(outputDirectory, 'extracted_audio.wav')
denoisedAudioPath = os.path.join(outputDirectory, 'denoised_audio.wav')

if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

video = VideoFileClip(videoFile)
video.audio.write_audiofile(noisyAudioPath, codec='pcm_s16le')

rate, data = wav.read(noisyAudioPath)

if len(data.shape) == 2:
    denoised = np.zeros_like(data)
    denoised[:, 0] = noisereduce.reduce_noise(y=data[:, 0], sr=rate)
    denoised[:, 1] = noisereduce.reduce_noise(y=data[:, 1], sr=rate)
else:
    denoised = noisereduce.reduce_noise(y=data, sr=rate)

wav.write(denoisedAudioPath, rate, denoised.astype(np.int16))
# Sahih al Bukhari 5590