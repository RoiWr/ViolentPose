''' Code to run for marking of Pose estimation joints and tracking of persons across video frames '''

import os
import pickle
import numpy as np
from joint_tracking import joint_tracking
import math
from scipy.ndimage.filters import gaussian_filter
import cv2
import util
import time
import argparse


def draw_boxes(input_image, frame_data, no_objects, colors):
    canvas = input_image.copy()
    for o in range(no_objects):
        # check if object i in frame
        if any(frame_data[:, -1] == o):
            object_bbox = tuple(list(frame_data[frame_data[:, -1].astype(int) == o, 1:5].astype(int).flatten()))
            # cv2.rectangle(image, start_point, end_point, color, thickness)
            canvas = cv2.rectangle(img=canvas, pt1=object_bbox[0:2], pt2=object_bbox[2:4],
                                   color=colors[o], thickness=3)
    return canvas

def draw_joints(input_image, all_peaks, subset, candidate, resize_fac=1):
    canvas = input_image.copy()

    for i in range(18):
        for j in range(len(all_peaks[i])):
            a = all_peaks[i][j][0] * resize_fac
            b = all_peaks[i][j][1] * resize_fac
            cv2.circle(canvas, (a, b), 2, util.colors[i], thickness=-1)

    stickwidth = 4

    for i in range(17):
        for s in subset:
            index = s[np.array(util.limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            y = candidate[index.astype(int), 0]
            x = candidate[index.astype(int), 1]
            m_x = np.mean(x)
            m_y = np.mean(y)
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(m_y * resize_fac), int(m_x * resize_fac)),
                                       (int(length * resize_fac / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, util.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def color_generator(n):
    colors = []
    for i in range(n):
        colors.append((np.random.randint(255), np.random.randint(255), np.random.randint(255)))
    return colors

def create_video(video_path, save_dir_path, joints, tracks):
    frame_rate_ratio = 3
    video = os.path.basename(video_path).split('.')[0]

    print(f'start processing video {video}')

    # Output location
    output_path = save_dir_path + '/'
    output_format = '.mp4'
    video_output = output_path + video + '_pose_tracking' + output_format

    # Video reader
    cam = cv2.VideoCapture(video_path)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, output_fps, (orig_image.shape[1], orig_image.shape[0]))

    # object (person) tracking
    no_objects = int(tracks[:, -1].max()) + 1
    colors = color_generator(no_objects)

    i = 0  # input video frame id
    j = 0  # analyzed frames id
    while (cam.isOpened()) and ret_val is True and i < ending_frame:
        if i % frame_rate_ratio == 0:
            print('Processing frame: ', i)
            input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

            tic = time.time()

            # generate image with body parts
            body_parts = joints[j]['body_parts']
            all_peaks = joints[j]['all_peaks']
            subset = joints[j]['subset']
            candidate = joints[j]['candidate']

            canvas = draw_joints(orig_image, all_peaks, subset, candidate)

            # draw bounding boxes based on data analyzed by joint_tracking.py
            frame_data = tracks[tracks[:, 0].astype(int) == i, :]
            try:
                canvas = draw_boxes(canvas, frame_data, no_objects, colors)
            except Exception as e:
                print(f'Error occurred in drawing boxes in frame {i}')
                print(e)

            out.write(canvas)

            toc = time.time()
            print('processing time is %.5f' % (toc - tic))

            j += 1
        ret_val, orig_image = cam.read()

        i += 1

    print(f'Finished processing video {video}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='input video name')
    parser.add_argument('--joints', type=str, help='input joints data (pkl)')
    parser.add_argument('--tracks', type=str, help='input tracking data (csv)')
    parser.add_argument('--video_dir', type=str, default='',
                        help='input video directory name')
    parser.add_argument('--data_dir', type=str, default='',
                        help='input data directory name')
    parser.add_argument('--save_dir', type=str, default='',
                        help='output data directory name')
    args = parser.parse_args()

    video_name = args.video
    joints_path = args.joints
    tracks_path = args.tracks
    data_dir = args.data_dir
    save_dir = args.save_dir
    video_dir = args.video_dir
    video_path = os.path.join(video_dir, video_name)

    print(f"Process file {video_name}")

    # read data
    filepath = os.path.join(data_dir, joints_path)
    with open(filepath, 'rb') as file:
        joints = pickle.load(file)

    filepath = os.path.join(data_dir, tracks_path)
    tracks = np.genfromtxt(filepath, delimiter=',')

    # mark videos and save them
    create_video(video_path, save_dir, joints, tracks)

def tg():
    video_path = 'sample_data/v_2VPd5Hcraxw.mp4'
    save_dir = 'sample_data'
    data_file_path = 'sample_data/v_2VPd5Hcraxw.pkl'
    with open(data_file_path, 'rb') as file:
        joints = pickle.load(file)

    tracks = joint_tracking(joints, person_thresh=0)
    # mark videos and save them
    create_video(video_path, save_dir, joints, tracks)


if __name__ == '__main__':
    tg()
