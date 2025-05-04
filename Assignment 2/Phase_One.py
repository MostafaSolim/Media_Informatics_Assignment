import cv2
import numpy as np
import os
import matplotlib.pyplot as plt  # Import matplotlib

Video = 'video_with_letters_precise (3).mp4'
outputDirectory = 'motion_frames'
MOTION_THRESHOLD = 0.20 # this is the percentage

if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

vid = cv2.VideoCapture(Video)
ret, prev_frame = vid.read()

frame_count = 0
unique_count = 0

first_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, curr_frame = vid.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(first_gray, curr_gray)
    diff_enhanced = cv2.equalizeHist(diff)

    _, binary = cv2.threshold(diff_enhanced, 30, 255, cv2.THRESH_BINARY)

    motion_ratio = np.sum(binary == 255) / binary.size

    if motion_ratio <= MOTION_THRESHOLD:
        filename = os.path.join(outputDirectory, f'frame_{frame_count:04d}.png')
        cv2.imwrite(filename, binary)
        unique_count += 1

        plt.imshow(binary, cmap='gray')
        plt.title(f'Frame {frame_count}')
        plt.axis('off')  # Hide axes
        plt.show()

    first_gray = curr_gray.copy()
    frame_count += 1

vid.release()
print(f"Processed {frame_count} frames, saved {unique_count} clear frames to '{outputDirectory}'")
