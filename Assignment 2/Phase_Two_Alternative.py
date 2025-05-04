import subprocess
import os
from pydub import AudioSegment
from pydub.effects import normalize
import noisereduce as nr
from scipy.io import wavfile
import numpy as np

# Input/Output files
video = "Fruit Animation.mp4"
extracted_wav = "extracted_audio.wav"
denoised_out = "denoised_audio.wav"

# 1. First check if FFmpeg is available
try:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except (subprocess.CalledProcessError, FileNotFoundError):
    raise RuntimeError("FFmpeg is not installed or not in system PATH. Please install FFmpeg first.")

# 2. Check if input video exists
if not os.path.exists(video):
    raise FileNotFoundError(f"Input video file not found: {video}")

# 3. Extract audio using FFmpeg
try:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i", video,
        "-vn",  # disable video
        "-acodec", "pcm_s16le",  # audio codec
        "-ar", "44100",  # sample rate
        "-ac", "2",  # stereo audio
        extracted_wav
    ]

    # For PyCharm, you might need to specify the full path to ffmpeg
    # ffmpeg_cmd[0] = "C:/path/to/ffmpeg.exe"  # Uncomment and set your path if needed

    res = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
except subprocess.CalledProcessError as e:
    print("FFmpeg command failed:")
    print(e.stderr.decode())
    raise

# 4. Process the extracted audio
try:
    # Read WAV file
    rate, data = wavfile.read(extracted_wav)

    # Convert to mono and float32
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)

    # Get noise profile (first 0.5 seconds)
    noise_samples = int(0.5 * rate)
    noise_clip = data[:noise_samples] if len(data) > noise_samples else data[:len(data) // 2]

    # Apply noise reduction
    reduced_noise = nr.reduce_noise(
        y=data,
        y_noise=noise_clip,
        sr=rate,
        stationary=False,
        prop_decrease=0.95
    )

    # Normalize and convert back to int16
    reduced_noise = np.int16(reduced_noise * (32767 / np.max(np.abs(reduced_noise))))

    # Save denoised audio
    wavfile.write(denoised_out, rate, reduced_noise)

    # Normalize with pydub
    audio = AudioSegment.from_wav(denoised_out)
    normalized = normalize(audio)
    normalized.export(denoised_out, format="wav")

    print("Audio processing completed successfully!")
    print(f"Denoised audio saved to: {os.path.abspath(denoised_out)}")

except Exception as e:
    print(f"Error during audio processing: {str(e)}")
    raise