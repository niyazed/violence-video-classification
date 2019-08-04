from vtf import *
import os

path="E:/Niloy/Journal/REVIEW WORK/video/violent/"

output_location="E:/Niloy/Journal/REVIEW WORK/frames/violent/"

videos = [vfile for vfile in os.listdir(path)]

for video in videos:
    video_path=os.path.join(path,video)
    print("Current Videofile Name: ",video)
    video_to_frames(video_path,output_location,video)





