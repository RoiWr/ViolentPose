''' Code to run for parsing of Pose estimation output and tracking of persons accross video frames '''

import os
import pickle

from sklearn.metrics import pairwise_distances_argmin
from sort.sort import *
import numpy as np

# CONSTANTS
COCO_BODY_PARTS = ['nose', 'neck',
                   'right_shoulder', ' right_elbow', 'right_wrist',
                   'left_shoulder', 'left_elbow', 'left_wrist',
                   'right_hip', 'right_knee', 'right_ankle',
                   'left_hip', 'left_knee', 'left_ankle',
                   'right_eye', 'left_eye', 'right_ear', 'left_ear', 'background'
                   ]


def get_person_pose_frame(person, candidate, joint_thresh=0):
    ''' extracts body positions for given person (item in the "subset" list) '''
    no_joints = person[19]
    config_score = person[18]  # check configuration score
    joints = person[0:18]
    joints_locs = np.zeros((18, 2))
    for i, joint in enumerate(joints):
        j = int(joint)
        if joint == -1 or j >= len(candidate):
            joints_locs[i, :] = [-1, -1]
        elif candidate[j, 2] < joint_thresh:
            joints_locs[i, :] = [-1, -1]
        else:
            joints_locs[i, :] = candidate[j, 0:2]

    return joints_locs, config_score, no_joints

def is_person(person, person_thresh=15):
    config_score = person[18]
    if config_score < person_thresh:
        return False
    else:
        return True

def get_bbox_from_pose(joints_locs):
    joints_locs_min = joints_locs.copy()
    joints_locs_min[joints_locs <= 0] = np.inf

    minx = min(joints_locs_min[:, 0])
    miny = min(joints_locs_min[:, 1])
    maxx = max(joints_locs[:, 0])
    maxy = max(joints_locs[:, 1])
    return [minx, miny, maxx, maxy]

def get_bbox_center(bbox):
    '''Returns the x, y coordinates of the center of a
    bounding box of the following format [x1,y1,x2,y2] (opposite corners).
    bbox = array of bounding boxes '''
    if len(bbox.shape) == 1:
        bbox = bbox.reshape(1, -1)
    x = np.mean(bbox[:,[0, 2]], axis=1)
    y = np.mean(bbox[:,[1, 3]], axis=1)
    return np.column_stack((x, y))

def get_all_joints(data, person_thresh=15, joint_thresh=0):
    '''Function outputs an array of detected persons in each frame ,
    their joint body parts x, y locations and bounding box [x1,y1,x2,y2]'''
    joints_array = []
    for i, frame_data in enumerate(data):
        frame_id = frame_data['frame_id']
        # print(f'Processing frame {i}')
        subset = np.array(frame_data['subset'])
        candidate = np.array(frame_data['candidate'])
        for person in subset:
            if not is_person(person, person_thresh=person_thresh):
                continue
            joints_locs, confidence, no_joints = get_person_pose_frame(person, candidate, joint_thresh=joint_thresh)
            bboxes = get_bbox_from_pose(joints_locs)
            joints_array.append([frame_id] + list(bboxes) + list(joints_locs.flatten()) + [confidence, no_joints])
    return np.array(joints_array)

def track_persons(joints_array, n_frames, frame_rate):
    ''' track the identified persons from different frames by the SORT algorithm.
    :return joints_array with new column of object_id '''
    # create instance of SORT
    mot_tracker = Sort()

    object_ids = []
    for i in np.arange(0, n_frames, frame_rate):
        detections = joints_array[joints_array[:, 0] == i, 1:5]
        if detections.size == 0:
            continue
        detections[detections == np.inf] = 100000
        # valid_dets_idx = np.argwhere(~np.isinf(detections).any(axis=1))[0]

        # update SORT l
        # trackers is a np array where each row contains a valid bounding box and track_id (last column)
        trackers = mot_tracker.update(detections)
        valid_trks_idx = np.argwhere(~np.isnan(trackers).any(axis=1)).reshape(-1)
        if trackers.size == 0 or valid_trks_idx.size == 0:
            object_ids_frame = np.tile(-1, detections.shape[0])
            object_ids += list(object_ids_frame)
            continue

        # match detections and trackers based on bbox centers distance
        det_centers = get_bbox_center(detections)
        trk_centers = get_bbox_center(trackers[valid_trks_idx, 0:4])
        indexes = pairwise_distances_argmin(det_centers, trk_centers, axis=0, metric='euclidean')
        object_ids_frame = np.tile(-1, detections.shape[0])
        object_ids_frame[indexes] = trackers[valid_trks_idx, -1]
        object_ids += list(object_ids_frame)

    # concatenate object ids as new column to joints_array
    object_ids = np.array(object_ids)
    object_ids[object_ids >= 0] = object_ids[object_ids >= 0] - min(object_ids[object_ids >= 0])  # reindex from 0
    return np.column_stack((joints_array, object_ids))

def joint_tracking(data, person_thresh=0, joint_thresh=0):
    array1 = get_all_joints(data, person_thresh=person_thresh, joint_thresh=joint_thresh)
    frame_rate = int(data[1]['frame_id'] - data[0]['frame_id'])
    n_frames = len(data) * frame_rate
    return track_persons(array1, n_frames, frame_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='input file name')
    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\Roi\Documents\ITC\PoseProject\out\n_segmented\data',
                        help='input data directory name')
    parser.add_argument('--save_dir', type=str,
                        default=r'C:\Users\Roi\Documents\ITC\PoseProject\out\n_segmented\tracks',
                        help='output data directory name')
    parser.add_argument('--person_thresh', type=int, default=0,
                        help='configuration score of person threshold')
    parser.add_argument('--joint_thresh', type=int, default=0,
                        help='joint detection threshold')
    args = parser.parse_args()

    data_dir = args.data_dir
    filename = args.file
    save_dir = args.save_dir
    person_thresh = args.person_thresh
    joint_thresh = args.joint_thresh

    print(f"Process file {filename}")

    # read file
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    frame_rate = int(data[1]['frame_id'] - data[0]['frame_id'])
    n_frames = len(data) * frame_rate
    array1 = get_all_joints(data, person_thresh=person_thresh, joint_thresh=joint_thresh)
    array2 = track_persons(array1, n_frames, frame_rate)

    # save array2 as csv
    coco_parts_xy = [[part + '_x', part + '_y'] for part in COCO_BODY_PARTS[0:-1]]
    array_header = ['frame_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'] +\
                   [l.strip(' ') for p in coco_parts_xy for l in p] + \
                   ['config_score', 'no_joints', 'object_id']
    out_filename = filename.split('.')[0] + '_data.csv'
    out_filepath = os.path.join(save_dir, out_filename)
    np.savetxt(out_filepath, array2, header=",".join(array_header), delimiter=",")
    print(f"File {filename} successfully analyzed. Saved to {out_filepath}")
