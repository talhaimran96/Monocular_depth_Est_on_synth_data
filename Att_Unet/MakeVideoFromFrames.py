import os
import cv2
import numpy as np
from tqdm import tqdm

# Output directory for saved frames
input_directory = "../test_results/SwinUnet_12eps_signmoid"

# Collect frames from the saved directory
frames = []
for filename in sorted(os.listdir(input_directory)):
    if filename.endswith(".jpg"):
        frame_path = os.path.join(input_directory, filename)
        frame = cv2.imread(frame_path)
        frames.append(frame)

# Specify the output video path
output_video_path = "../test_results/SwinUnet_12eps_signmoid/output_video.mp4"

# Get frame dimensions
height, width, _ = frames[0].shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video_path, fourcc, 1.0, (width, height))

# Write frames to video
for frame in tqdm(frames, desc="Creating Video"):
    video_writer.write(frame)

# Release the video writer
video_writer.release()

print(f"Video saved at: {output_video_path}")
