'''Script to cut the downloaded clips of the HACS dataset to the annotated labeled 2 sec clips
Roi Weinberger & Sagiv Yaarri. Date: 1/2/2020 '''

import os
import csv
import pandas as pd
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# CONSTANTS
ANNOTATION_FILE_PATH = '/data/smoking_pose/HACS/HACS-dataset/HACS_v1.1.1/HACS_clips_v1.1.1.csv'
VIDEO_DIR = '/data/smoking_pose/HACS'
SAVE_DIR = '/data/smoking_pose/in'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, help='category of videos to cut to clips')

    args = parser.parse_args()
    category = args.category

    # load annotations csv and subset to category
    df = pd.read_csv(ANNOTATION_FILE_PATH)
    df_cat = df.loc[df.classname == category, :]
    del df

    # create save folder
    category_dir = category.replace(' ', '_')
    
    # labels file
    labels_path = os.path.join(SAVE_DIR, 'labels.csv')
    if not os.path.isfile(labels_path):
        with open('names.csv', 'w', newline='') as csvfile:
            fieldnames = df_cat.columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
    for i, row in df_cat.iterrows():
        filename = f'v_{row.youtube_id}.mp4'
        filepath = os.path.join(VIDEO_DIR, category_dir, filename) 
        outfilename = f'v_{row.youtube_id}_{row.start:.0f}_{row.end:.0f}.mp4'
        outfile_path = os.path.join(SAVE_DIR, outfilename)
        
        if os.path.isfile(filepath) and not os.path.isfile(outfile_path):
            try:
                ffmpeg_extract_subclip(filepath, row['start'], row['end'], targetname=outfile_path)
            except Exception as e:
                print(f'Error: No. {i}: Video {row.youtube_id}. skip to next video')
                print(e)
                continue
                
            # appending to label data csv
            with open(labels_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(row.values.astype(str)))  
                
            print(f'No. {i}: Clipped video {filename} successfully')
            
        else:
            print(f'No. {i}: Video {row.youtube_id} not found or exists. skip to next video')
            continue

