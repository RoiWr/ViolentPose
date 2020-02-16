'''Script to cut the downloaded clips of the HACS dataset to 10 seconds clips from segments dataset
Roi Weinberger & Sagiv Yaarri. Date: 5/2/2020 '''

import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import json


# CONSTANTS
ANNOTATION_FILE_PATH = '/data/smoking_pose/HACS/HACS-dataset/HACS_v1.1.1/HACS_segments_v1.1.1.json'
MAX_CLIP_LENGTH = 10  #seconds

VIDEO_DIR = '/data/smoking_pose/HACS/'

out_dir = os.path.join(VIDEO_DIR,'segmented')
print(out_dir)
with open(ANNOTATION_FILE_PATH, 'r') as f:
    segments = json.loads(f.read())
    segments = segments['database']


categories = os.listdir(VIDEO_DIR)

labels = [list(set(ann['label'] for ann in segments[vid]['annotations'])) for vid in segments]
labels = list(set([lab[0].replace(' ','_') for lab in labels if len(lab)==1]))

categories = [cat for cat in categories if cat in labels]

for category in categories:
    video_names = os.listdir(os.path.join(VIDEO_DIR,category))
    segmented_video_dir = os.path.join(VIDEO_DIR,'segmented')
    if not os.path.isdir(segmented_video_dir):
        os.mkdir(segmented_video_dir)
    for vid in video_names:
        if vid[2:-4] in segments.keys():
            vid_annotations = segments[vid[2:-4]]['annotations']
            category = vid_annotations[0]['label'].replace(' ','_')
            out_dir = os.path.join(segmented_video_dir,category)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            print(vid)
            filepath = os.path.join(os.path.join(VIDEO_DIR,category),vid)
            vid_annotations = segments[vid[2:-4]]['annotations']
            for ann in vid_annotations:
                start = ann['segment'][0]
                end = min(ann['segment'][1],start+MAX_CLIP_LENGTH)
                try:
                    with VideoFileClip(filepath) as video:
                        new = video.subclip(start, end)

                        new.write_videofile(os.path.join(out_dir,vid), audio_codec='aac')
                except Exception as ex:
                    print(ex)
        else:
            print(f'Couldn\'t find annotation data for video: {vid}')
        
