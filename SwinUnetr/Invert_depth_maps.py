import cv2
from glob import glob
import os
from tqdm import tqdm

path = "../../../Desktop/Renders/test"

for scenes in os.listdir(path):
    print(scenes)
    for file in tqdm(glob(f"{path}/{scenes}/Depth_maps/*.png")):
        img = 255 - cv2.imread(file)
        cv2.imwrite(file, img)
