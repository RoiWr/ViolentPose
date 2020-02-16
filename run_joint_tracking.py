import os
import glob
import pickle
import numpy as np
from joint_tracking import joint_tracking
from video_marking import create_video


# VIDEO_DIR = r'C:\Users\Roi\Documents\ITC\PoseProject\in\clips'
# DATA_DIR = r'C:\Users\Roi\Documents\ITC\PoseProject\out\data'
# SAVE_DIR = r'C:\Users\Roi\Documents\ITC\PoseProject\out\tracks'
# VIDEO_SAVE_DIR = r'C:\Users\Roi\Documents\ITC\PoseProject\out\tracked_vids'
VIDEO_DIR = r'C:\Users\Roi\Documents\ITC\PoseProject\in\segmented'
DATA_DIR = r'C:\Users\Roi\Documents\ITC\PoseProject\out\n_segmented\data'
SAVE_DIR = r'C:\Users\Roi\Documents\ITC\PoseProject\out\n_segmented\tracks'
VIDEO_SAVE_DIR = r'C:\Users\Roi\Documents\ITC\PoseProject\out\n_segmented\tracked_vids'
# VIDEO_DIR = r'C:\Users\Roi\Documents\ITC\SmokingPose\sample_data\debug_track'
# DATA_DIR = r'C:\Users\Roi\Documents\ITC\SmokingPose\sample_data\debug_track'
# SAVE_DIR = r'C:\Users\Roi\Documents\ITC\SmokingPose\sample_data\debug_track'
# VIDEO_SAVE_DIR = r'C:\Users\Roi\Documents\ITC\SmokingPose\sample_data\debug_track'
COCO_BODY_PARTS = ['nose', 'neck',
                   'right_shoulder', 'right_elbow', 'right_wrist',
                   'left_shoulder', 'left_elbow', 'left_wrist',
                   'right_hip', 'right_knee', 'right_ankle',
                   'left_hip', 'left_knee', 'left_ankle',
                   'right_eye', 'left_eye', 'right_ear', 'left_ear', 'background'
                   ]

if __name__ == '__main__':
    video_list = glob.glob(VIDEO_DIR + r'/*.mp4')
    filelist = glob.glob(DATA_DIR + r'/*.pkl')
    for filename in filelist:
        print(f"Process file {filename}")
        try:
            with open(filename, 'rb') as f:
                joints = pickle.load(f)
        except Exception as e:
            print(e)
            continue

        # try:
        tracks = joint_tracking(joints)
        # except Exception as e:
        #     print(f"Error is joint_tracking file {filename}")
        #     print(e)
        #     continue

        # save to csv
        coco_parts_xy = [[part + '_x', part + '_y'] for part in COCO_BODY_PARTS[0:-1]]
        array_header = ['frame_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'] + \
                       [l.strip(' ') for p in coco_parts_xy for l in p] + \
                       ['config_score', 'no_joints', 'object_id']
        out_filename = os.path.basename(filename).split('.')[0] + '_data.csv'
        out_filepath = os.path.join(SAVE_DIR, out_filename)
        np.savetxt(out_filepath, tracks, header=",".join(array_header), delimiter=",")
        print(f"File {filename} successfully analyzed. Saved to {SAVE_DIR}")

        # write video

        out_video_filename = os.path.basename(filename).split('.')[0]
        if out_video_filename in '.'.join(video_list):
            video_path = os.path.join(VIDEO_DIR, out_video_filename + '.mp4')
            try:
                create_video(video_path, VIDEO_SAVE_DIR, joints, tracks)
                print(f"Successfully saved tracked video for {out_video_filename}")
            except Exception as e:
                print(f"Error in create_video for {out_video_filename}")
                print(e)