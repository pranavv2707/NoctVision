import cv2
import numpy as np
from pathlib import Path

def meanfinder(video_id,folder="data/processed/images/train",step=15):
    all_frames= sorted(Path(folder).glob(f"{video_id}*.jpg")) #thats the complete path of the image file
    sampledframes= all_frames[::step]
    values=[]
    for i in sampledframes:
        image = cv2.imread(str(i),cv2.IMREAD_GRAYSCALE) #first grayscale conversion
        values.append(image.mean())
    return np.mean(values),np.std(values), len(all_frames)

potentialvideos=["01","02","03","04","05","06","07","11","12","13","14","15","16","17","18","25"]

for i in potentialvideos:
    mean,stddev,frames=meanfinder(i)
    print(f"for video {i} the mean is: {mean}, standard deviation is: {stddev} and the length is {frames}")

    
